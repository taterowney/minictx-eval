# Evaluation script for miniCTX dataset
# Written by Tate Rowney and Jiewen Hu (CMU L3 lab)

import json, os, re, subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tqdm

from agent_boilerplate import Client, get_model_vendor
from repl_wrapper import evaluate_repl, InteractiveThread


MINICTX_PATH = os.path.join(os.getcwd(), "data/minictx/")
MINICTX2_PATH = os.path.join(os.getcwd(), "data/minictx2/test/")


def _prompt_fewshot(theorem_statement, src_context, task="full_proof_context", premises=None):
    """ Generates a prompt with a few examples for the theorem proving task. """
    if premises is not None and task.endswith("premise"):
        with open(f"prompt/{task}_premise.txt", "r") as infile:
            prompt = infile.read()
        premises_str = "\n".join(premises)
        prompt = prompt.format(src_context, premises_str, theorem_statement)

    else:
        with open(f"prompt/{task}.txt", "r") as infile:
            prompt = infile.read()
        if task.endswith("context") or task.endswith("repository"):
            prompt = prompt.format(src_context, theorem_statement)
        else:
            prompt = prompt.format(theorem_statement)

    return prompt


def _get_proof_delimiters(task):
    """ Returns the delimiters used by this task to contain the output proof. """
    if task.startswith("full_proof"):
        return "```lean", "```"
    return "[TAC]", "[/TAC]"


def _load_system_prompt():
    """ Loads the system prompt from the prompts/ directory. """
    with open("prompt/system_prompt.txt", "r") as infile:
        return infile.read()


def _extract_proof(response, theorem_statement, task):
    """ Extracts the contents of the first block in the proof in the relevant format.
     ENSURES: does NOT contain theorem statement """
    # print(response)
    delimiters = _get_proof_delimiters(task)
    # pattern = re.compile(r'```lean(.*?)```', re.DOTALL | re.IGNORECASE)
    pattern = re.compile(rf'{re.escape(delimiters[0])}(.*?){re.escape(delimiters[1])}', re.DOTALL | re.IGNORECASE)
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
        raise ValueError(f"{dataset_name} is not a miniCTX-v2 dataset, so can't automatically determine which environment to use. Please specify one using the --lean-env-path argument.")
    dir = os.path.join(os.getcwd(), envs[dataset_name.lower()])
    if not os.path.exists(dir):
        raise ValueError(f"Lean environment directory {dir} is not installed. Run `git submodule init` and `git submodule update`, the go to {dir} and run `lake exe cache get` and `lake build` to install it.")
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
    """ Returns a path for the output file to contain the results of the evaluation. """
    output_dir = os.path.join(os.getcwd(), "output", dataset_name, datetime.now().strftime("%d-%m-%Y-%H-%M")+f":{model}@{num_samples}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results.jsonl")
    return output_file


def _truncate_middle(text, max_tokens, tokenizer=None):
    """
    Truncates the middle of the text to fit within the max_tokens limit.
    Since most of minictx is in English, we have around 4 characters per token on average (https://platform.openai.com/tokenizer), with pretty minor variation between model vendors.
    """
    if tokenizer is None:
        tokens = text.split()
        if len(tokens) <= max_tokens*4:
            return text
        half = max_tokens * 2 - 1
        return ' '.join(tokens[:half] + ["..."] + tokens[-half:])
    else:
        # Use the tokenizer to count tokens
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        half = max_tokens // 2 - 1
        truncated_tokens = tokens[:half] + tokenizer.encode("...") + tokens[-half:]
        return tokenizer.decode(truncated_tokens)


def _check_lake():
    """ Checks if the lake executable is available in the PATH. """
    if not subprocess.run(["which", "lake"], capture_output=True, text=True, check=True):
        raise EnvironmentError("Lake executable not found in PATH. Make sure Lean (and its package manager `lake`) are installed. If you are running this in an IDE, try running via the command line instead. ")


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


def verify_fn(candidates, context, theorem_statement, repl_path, lean_env_path):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_repl, context, theorem_statement + candidate, repl_path=repl_path, lean_env_path=lean_env_path) for candidate in candidates]
        results = [future.result() for future in futures]
    return [{"status": "done", "output": res} if res["success"] else ({"status": "valid"} if res["errors"].startswith("unsolved goals") else {"status": "invalid"}) for res in results]

def sampling_eval(client):
    def _inner(proofs, theorem_statement, context):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                client.get_response,
                _prompt_fewshot(theorem_statement + proof, context, task="tactic_prediction_context")
            ) for proof in proofs]
            return [_extract_proof(f.result(), theorem_statement, task="tactic_prediction_context") for f in futures]
    return _inner

#TODO: Pantograph
def sampling_search(example, k, eval_fn, max_iters, repl_path, lean_env_path):
    """ Performs a tactic-by-tactic sampling search for the proof of a theorem statement. """
    proofs = ["" for _ in range(k)]  # Initialize k empty proofs
    context = example.get("srcContext", "")
    theorem_statement = example["theoremStatement"].strip() + " := by\n" # Tactic mode!

    for _ in range(max_iters):
        candidates = eval_fn(proofs, theorem_statement, context)
        print(candidates)
        candidates = [proofs[i] + candidate for i, candidate in enumerate(candidates)]

        for i, (candidate, res) in enumerate(zip(candidates, verify_fn(candidates, context, theorem_statement, repl_path, lean_env_path))):
            if res["status"] == "done":
                return {"success": True, "proof": candidate}
            elif res["status"] == "valid":
                proofs[i] += candidate + "\n"

    return {"success": False, "proof": None}


def evaluate_full_proofs(client, data, repl_path, lean_env_path, task="full_proof_context", dataset_name="mathlib", n=32, batch=False):

    prompts = []
    for example in data:
        theorem_statement = example["theoremStatement"].strip() + " := "
        if task == "full_proof_repository":
            from load_repository import load_repository_context
            context = load_repository_context(lean_env_path, example["file"], theorem_statement)
        else:
            context = example.get("srcContext", "")
        prompt = _prompt_fewshot(theorem_statement, context, task=task)
        prompt = _truncate_middle(prompt, client.context_window, client.tokenizer)
        prompts.append(prompt)

    system_prompt = _load_system_prompt()

    if not batch:
        with ThreadPoolExecutor() as executor:
            futures = []
            for prompt in prompts:
                futures.append(executor.submit(
                    client.get_raw_response,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    n=n
                ))

            responses = [f.result() for f in tqdm.tqdm(futures, desc=f"Evaluating {len(prompts)} proofs with {client.model_name}")]

    else:
        prompts = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in prompts]
        responses = client.infer_batch(prompts, batch_name=dataset_name, n=n, job_tags={"task": task})

    ret = []
    for example, response in zip(data, tqdm.tqdm(responses, desc=f"Checking proofs for {dataset_name} generated by {client.model_name}")):
        theorem_statement = example["theoremStatement"].strip() + " :="
        context = example.get("srcContext", "")

        proofs = _unique(_extract_proof(res.message.content, theorem_statement, task) for res in response.choices)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_repl, context, theorem_statement + p, repl_path=repl_path, lean_env_path=lean_env_path) for p in proofs]
            results = [future.result() for future in futures]

        has_proven = False
        proof_data = {}
        for proof, result in zip(proofs, results):
            if result["success"]:
                has_proven = True
                proof_data = {"success": True, "proof": proof}
        if not has_proven:
            proof_data = {"success": False, "proof": None}

        example["full_name"] = _get_full_name(theorem_statement)
        example["proof"] = proof_data
        example["candidates"] = [res.message.content for res in response.choices]
        ret.append(example)

    return ret



def evaluation_loop(model_name, task="full_proof_context", dataset_name="mathlib", dataset_path=MINICTX2_PATH, log_output=True, output_dir=None, num_samples=32, repl_path=os.path.join(os.getcwd(), "repl"), lean_env_path=None, use_batch_inference=False, vllm_mode="offline"):
    """ Loads a dataset, evaluates the model on each task in it, and evaluates responses. """

    _check_lake()

    # Initialize a client (this will handle local vs API models)
    source = get_model_vendor(model_name)
    if source == "openai":
        source = "azure"
    if source == "vllm" and vllm_mode == "online":
        source = "vllm-online"
    if source != "local":
        context_window = 195000 # A good general minimum for most recent frontier models
    else:
        context_window = 128000
    client = Client(
        model_name=model_name,
        model_source=source,
    )
    client.context_window = context_window
    if source == "azure":
        import tiktoken
        client.tokenizer = tiktoken.get_encoding("o200k_base")
    else:
        client.tokenizer = None


    # Set the path to the Lean environment based on the dataset if not provided
    if lean_env_path is None:
        lean_env_path = _get_lean_environment_name(dataset_name)

    data = load_data(dataset_name, path_to_data=dataset_path)


    if task.startswith("full_proof"):
        results = evaluate_full_proofs(
            client,
            data,
            repl_path=repl_path,
            lean_env_path=lean_env_path,
            task=task,
            dataset_name=dataset_name,
            n=num_samples,
            batch=use_batch_inference
        )

    elif task.startswith("tactic_prediction"):
        results = []
        for example in tqdm.tqdm(data, desc=f"Evaluating {dataset_name} with {model_name}"):
            theorem_statement = example["theoremStatement"].strip() + " := "
            context = example.get("srcContext", "")

            def eval_parallel(proofs_so_far):
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(
                        client.get_response,
                        _prompt_fewshot(theorem_statement + p, context, task=task),
                        n=num_samples
                    ) for p in proofs_so_far]
                    return [_extract_proof(f.result(), theorem_statement, task) for f in futures]

            results.append(
                sampling_search(
                    example,
                    k=num_samples,
                    # eval_fn=lambda proofs: [_extract_proof(client.get_response(_prompt_fewshot(theorem_statement + p, context, task)), theorem_statement, task) for p in proofs],
                    eval_fn=eval_parallel,
                    max_iters=10,
                    repl_path=repl_path,
                    lean_env_path=lean_env_path,
                )
            )

    successes = 0
    for res in results:
        # print(res)
        if res["proof"]["success"]:
            successes += 1

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
        # choices=['tactic_prediction', 'tactic_prediction_context', 'full_proof', 'full_proof_context', 'tactic_prediction_context_premise', 'full_proof_context_premise'],
        choices=['full_proof', 'full_proof_context', 'full_proof_repository'],
        help="The task for the model to perform. 'tactic_prediction' is for predicting the next tactic based on the theorem statement, 'full_proof' attempts to generate a full proof, while the '..._context' options provide the full context of the repository to the model as well."
    )
    parser.add_argument(
        '--dataset-name',
        default='mathlib',
        help="The dataset to evaluate the model on. MiniCTX-v2 includes mathlib, carleson, ConNF, FLT, foundation, HepLean, and seymour. You may specify another dataset by using this and the --dataset-path flag"
    )
    parser.add_argument('--dataset-path', default=MINICTX2_PATH, help="The path to the dataset directory. Defaults to miniCTX-v2 test set. The path specified should be a directory containing appropriately formatted .jsonl files (see data/minictx2/test/ for examples), while the --dataset-name argument should be the name of one of the jsonl files. ")
    # parser.add_argument('--premise-path', default=None)
    parser.add_argument('--log-output', type=bool, default=True, help="Whether to log the output to a file.")
    parser.add_argument('--output-dir', default=None, help="The directory to save the output to. If not specified, a new directory will be created in the output/{dataset name} folder with the current date and time.")
    # parser.add_argument('--tp-degree', type=int, default=1)
    # parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=8, help="The number of samples to generate for each theorem (best of N).")
    # parser.add_argument('--temperatures', type=float, default=0.0)
    #TODO: automatically select correct REPL path based on dataset name
    parser.add_argument('--repl-path', default=os.path.join(os.getcwd(), "repls/repl-4.16.0"), help="The path to the REPL submodule, used to evaluate the model's proofs.")
    parser.add_argument('--lean-env-path', default=None, help="The path to the Lean environment to use for evaluating proofs. If not specified, it will be automatically determined based on the dataset name.")
    parser.add_argument('--use-batch-inference', type=bool, default=False, help="Whether to use batch inference for the model. May not be supported by all models. For API models, this generally takes a very long time, but is more cost-effective.")
    parser.add_argument("--vllm-mode", default="offline", choices=["offline", "online"], help="When using vLLM, whether to use online or offline inference. Online inference requires a vLLM server to be running (vllm serve --port 8000 --model <model_name>). Offline inference does not support reasoning models.")

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
        vllm_mode=args.vllm_mode
    )

    # python3 check_new.py --model-name "gemini-2.5-flash-preview-05-20" --dataset-name "mathlib"
    # python3 check_new.py --model-name "gemini-2.5-pro-preview-05-06" --dataset-name "mathlib"
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "mathlib" --num-samples 32 --use-batch-inference True

    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "carleson" --num-samples 8 --use-batch-inference True
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "ConNF" --num-samples 8 --use-batch-inference True --repl-path "repls/repl-4.16.0-rc1"
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "FLT" --num-samples 8 --use-batch-inference True
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "foundation" --num-samples 8 --use-batch-inference True
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "HepLean" --num-samples 8 --use-batch-inference True
    # python3 check_new.py --model-name "o4-mini-global-batch" --dataset-name "Seymour" --num-samples 8 --use-batch-inference True


    # Carleson: 18/50
    # ConNF: 15/50
    # FLT: 13/34
    # Foundation: 27/50
    # HepLean: 22/50
    # Seymour: 30/50
    # mathlib: 23/50
    #
    # All: 148/334 (44.3%)



    # python3 check_new.py --model-name "o4-mini-global-batch" --task="full_proof_repository" --dataset-name "carleson" --num-samples 8 --use-batch-inference True
    # python3 check_new.py --model-name "o4-mini-global-batch" --task="full_proof_repository" --dataset-name "con-nf" --num-samples 1 --use-batch-inference True
