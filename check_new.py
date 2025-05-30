import json, os, re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tqdm

from agent_boilerplate import Client, get_model_vendor
from repl_wrapper import evaluate_repl


MINICTX_PATH = os.path.join(os.getcwd(), "data/minictx/")
MINICTX2_PATH = os.path.join(os.getcwd(), "data/minictx2/test/")


def _prompt_fewshot(theorem_statement, src_context, task="full_proof_context", premises=None):
    """ Generates a prompt with a few examples for the theorem proving task. """
    if premises != None:
        with open(f"prompt/{task}_premise.txt", "r") as infile:
            prompt = infile.read()
    else:
        with open(f"prompt/{task}.txt", "r") as infile:
            prompt = infile.read()

    if task == "full_proof_context":
        if premises != None:
            premises_str = "\n".join(premises)
            prompt = prompt.format(src_context, premises_str, theorem_statement)
        else:
            prompt = prompt.format(src_context, theorem_statement)
        # TODO: truncation based on receiving model
        # prompt = truncate_middle(prompt, tokenizer, max_tokens=2000000)
        return prompt


def _load_system_prompt():
    """ Loads the system prompt from the prompts/ directory. """
    with open("prompt/system_prompt.txt", "r") as infile:
        return infile.read()


def _extract_proof(response, theorem_statement):
    """ Extracts the contents of the first "```lean ... ```" block in the proof.
     ENSURES: does NOT contain theorem statement """
    pattern = re.compile(r'```lean(.*?)```', re.DOTALL | re.IGNORECASE)
    match = re.search(pattern, response)
    if match:
        proof = match.group(1).strip()
        if proof.startswith(theorem_statement):
            proof = proof[len(theorem_statement):].strip()
        return proof
    else:
        return ""

def _get_lean_environment_name(dataset_name):
    """ Returns the directory of the Lean project for the specified dataset. """
    envs = {
        "mathlib": "test-envs/minictx-v2/mathlib4",
        "carleson": "test-envs/minictx-v2/carleson",
        "connf": "test-envs/minictx-v2/con-nf",
        "flt": "test-envs/minictx-v2/FLT",
        "foundation": "test-envs/minictx-v2/Foundation",
        "heplean": "test-envs/minictx-v2/Physlean", # note the name change
        "seymour": "test-envs/minictx-v2/seymour",
    }
    if dataset_name.lower() not in envs:
        raise ValueError(f"Can't find a directory with a Lean environment to run for dataset {dataset_name}. Please specify one using the --lean_env_path argument.")
    dir = os.path.join(os.getcwd(), envs[dataset_name.lower()])
    if not os.path.exists(dir):
        raise ValueError(f"Lean environment directory {dir} is not installed. Clone the {envs[dataset_name.lower()]} repository from GitHub and run `lake exe cache get` and `lake build` to install it.")
    return dir

def _unique(texts):
    """ Returns unique texts from the list, preserving order. Removes empty strings. """
    seen = set()
    unique_texts = []
    for text in texts:
        if text and (text not in seen):
            seen.add(text)
            unique_texts.append(text)
    return unique_texts

def _get_full_name(statement):
    """ Extracts the full name of the theorem or lemma from the statement. """
    word_list = statement.split()
    for i in range(len(word_list)):
        if "theorem" in word_list[i] or "lemma" in word_list[i]:
            return statement.split()[i+1]
    return None


def _get_output_file(dataset_name, model, num_samples):
    output_dir = os.path.join(os.getcwd(), "output", dataset_name, datetime.now().strftime("%d-%m-%Y-%H-%M")+f":{model}@{num_samples}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results.jsonl")
    return output_file


def _truncate_middle(text, max_tokens):
    """
    Truncates the middle of the text to fit within the max_tokens limit.
    Since most of minictx is in English, we have around 4 characters per token on average (https://platform.openai.com/tokenizer), with pretty minor variation between model vendors.
    """
    tokens = text.split()
    if len(tokens) <= max_tokens*4:
        return text
    half = max_tokens * 2 - 1
    return ' '.join(tokens[:half] + ["..."] + tokens[-half:])


def load_data(dataset_name, path_to_data=MINICTX2_PATH):
    """ Loads a dataset from a jsonl file in the specified directory. """
    to_file_name = {
        f.lower().split(".jsonl")[0]: f for f in os.listdir(path_to_data) if f.endswith(".jsonl")
    }
    if dataset_name.lower() not in to_file_name:
        raise ValueError(f"Dataset {dataset_name} not found in the directory {path_to_data}. Available datasets: {list(to_file_name.keys())}")
    data = []
    with open(os.path.join(path_to_data, to_file_name[dataset_name.lower()])) as f:
        for line in f.readlines():
            data_ = json.loads(line)
            data.append(data_)

    return data


# def evaluation_loop(dataset_name, model_name, task, dataset_path, prompt_fn=_prompt_fewshot, repl_path=os.path.join(os.getcwd(), "repl"), lean_env_path=None, log_output=True, output_dir=None):
def evaluation_loop(model_name, task="full_proof_context", dataset_name="mathlib", dataset_path=MINICTX2_PATH, log_output=True, output_dir=None, num_samples=32, repl_path=os.path.join(os.getcwd(), "repl"), lean_env_path=None, use_batch_inference=False, recheck_completed_job=False):
    """ Loads a dataset, evaluates the model on each task in it, and evaluates responses. """

    # Initialize a client (this will handle local vs API models)
    source = get_model_vendor(model_name)
    if source == "openai":
        source = "azure"
    if source != "local":
        context_window = 200000 # A good general minimum for most recent frontier models
    else:
        context_window = 128000
    client = Client(
        model_name=model_name,
        model_source=source,
    )

    # Set the path to the Lean environment based on the dataset if not provided
    if lean_env_path is None:
        lean_env_path = _get_lean_environment_name(dataset_name)

    data = load_data(dataset_name, path_to_data=dataset_path)

    # TODO: different based on task?
    prompt_fn = _prompt_fewshot

    # Evaluate examples in parallel...
    successes = 0
    responses = []
    if not use_batch_inference:
        with ThreadPoolExecutor() as executor:
            futures = []
            for example in data:
        # for example in tqdm.tqdm(data, desc=f"Evaluating {dataset_name} with {model_name}"):
                theorem_statement = example["theoremStatement"].strip() + " := "
                context = example.get("srcContext", "")
                prompt = prompt_fn(theorem_statement, context, task=task)
                prompt = _truncate_middle(prompt, context_window)

                system_prompt = _load_system_prompt()

                # ...get the model's responses...
                futures.append(executor.submit(
                    client.get_raw_response,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        n=num_samples
                    )
                )

            for f in tqdm.tqdm(futures, desc=f"Evaluating {dataset_name} with {model_name}"):
                response = f.result()
                print(response)
                responses.append(response)

    else:
        # Use batch inference for the model
        messages_list = []
        for example in data:
            theorem_statement = example["theoremStatement"] + " := "
            context = example.get("srcContext", "")
            prompt = prompt_fn(theorem_statement, context, task=task)
            prompt = _truncate_middle(prompt, context_window)

            system_prompt = _load_system_prompt()

            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

        responses = client.infer_batch(messages_list, resume_complete=recheck_completed_job, n=num_samples)

    # Ik it could probably be done faster by running it while inference happens, but I wanted to keep it consistent for batch inference too
    for example, response in zip(data, tqdm.tqdm(responses, desc=f"Checking proofs for {dataset_name} with {model_name}")):
        theorem_statement = example["theoremStatement"].strip() + " := "
        context = example.get("srcContext", "")
        # print(theorem_statement)

        proofs = _unique(_extract_proof(res.message.content, theorem_statement) for res in response.choices)
        # print(proofs)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_repl, context, theorem_statement + p, repl_path=repl_path, lean_env_path=lean_env_path) for p in proofs]
            results = [future.result() for future in futures]

        has_proven = False
        proof_data = {}
        for proof, result in zip(proofs, results):
            if result["success"]:
                successes += 1
                has_proven = True
                proof_data = {"success": True, "proof": proof}
                break
        if not has_proven:
            proof_data = {"success": False, "proof": proofs[0]} # Figured it would be useful to see where it's going wrong

        example["full_name"] = _get_full_name(theorem_statement)
        example["proof"] = proof_data

    print(f"""{'-'*20} SUMMARY {'-'*20}
Dataset: {dataset_name}
Model: {model_name}
Score: {successes} correct out of {len(data)} ({successes/len(data)*100:.2f}%)
{'-'*50}""")

    # Save results to a file
    if log_output:
        if output_dir is None:
            output_file = _get_output_file(dataset_name, model_name, num_samples)
        else:
            output_file = os.path.join(output_dir, f"results.jsonl")
        with open(output_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument(
        '--task',
        default='full_proof_context',
        choices=['tactic_prediction', 'tactic_prediction_context', 'full_proof', 'full_proof_context', 'tactic_prediction_fewshot']
    )
    parser.add_argument(
        '--dataset-name',
        default='mathlib'
    )
    parser.add_argument('--dataset-path', default=MINICTX2_PATH)
    # parser.add_argument('--premise-path', default=None)
    parser.add_argument('--log-output', type=bool, default=True)
    parser.add_argument('--output-dir', default=None)
    # parser.add_argument('--tp-degree', type=int, default=1)
    # parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=8)
    # parser.add_argument('--temperatures', type=float, default=0.0)
    parser.add_argument('--repl-path', default=os.path.join(os.getcwd(), "repl"))
    parser.add_argument('--lean-env-path', default=None)
    parser.add_argument('--use-batch-inference', type=bool, default=False)
    parser.add_argument("--recheck-completed-job", type=bool, default=False)

    args = parser.parse_args()

    evaluation_loop(
        model_name=args.model_name,
        task=args.task,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        log_output=args.log_output,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        repl_path=args.repl_path,
        lean_env_path=args.lean_env_path,
        use_batch_inference=args.use_batch_inference,
        recheck_completed_job=args.recheck_completed_job
    )

    # python3 check_new.py --model-name "gemini-2.5-flash-preview-05-20" --dataset-name "mathlib"
    # python3 check_new.py --model-name "gemini-2.5-pro-preview-05-06" --dataset-name "mathlib"
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "mathlib" --num-samples 32 --use-batch-inference True

