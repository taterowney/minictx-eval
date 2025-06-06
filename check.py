import json
import heapq
import subprocess
import os
import time
import transformers
# import vllm
import re
import sys
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm, trange

from repl_wrapper import InteractiveThread
from premise_retriever import retrieve

# from openai import AzureOpenAI
import tiktoken
import tempfile

from api_models import prompt_model, API_MODELS

os.system("source .env")
# openai.api_key = ""  # Fill openai API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "true"  

def _load_data(dataset_name, dataset_path):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
            for example in data:
                example["theoremStatement"] = example["statement"]
                del example["statement"]
                example["srcContext"] = "import MiniF2F.Minif2fImport\n  open BigOperators Real Nat Topology\n"
    else:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

    return data

def truncate_middle(text, tokenizer, max_tokens=8000):
    if tokenizer:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        keep_tokens = max_tokens // 2
        head_tokens = tokens[:keep_tokens]
        tail_tokens = tokens[-keep_tokens:]

        truncated_text = tokenizer.decode(head_tokens) + "..." + tokenizer.decode(tail_tokens)
        return truncated_text
    else:
        if len(text) <= max_tokens:
            return text
        return text[:max_tokens] + "..." + text[-max_tokens:]

def discard_after_marker(text, marker='/- BEGIN EXERCISES -/\n'):
    marker_index = text.find(marker)
    if marker_index != -1:
        return text[:marker_index] + marker
    return text

def _load_jsonl(filepath):  # return empty if file does not exist
    if os.path.exists(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    else:
        return []

def get_premises(example, premise_path):
    if not os.path.exists(premise_path):
        raise FileNotFoundError(f"The folder '{premise_path}' does not exist.")
    # folder = "pfr-declarations"
    project_premises = []
    for module in example["dependencyMetadata"]["importedModules"]:
        file = module + ".jsonl"
        premises = _load_jsonl(os.path.join(premise_path, file))
        project_premises += [premise["declaration"] for premise in premises]
    return project_premises

# def generate_api(prompt, model, temperatures, num_samples, max_tokens=256):
#     # print("Generating API response...")
#     texts, scores = [], []
#     azure_client = AzureOpenAI(
#         api_key = os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version = "2024-10-21",
#         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     )
#     for temperature in temperatures:
#         # Prepare API request parameters
#         # responses = openai.chat.completions.create(
#         #     model=model,
#         #     messages=[
#         #         {"role": "system", "content": "You are a helpful assistant who is an expert in the Lean theorem prover."},
#         #         {"role": "user", "content": prompt}
#         #     ],
#         #     max_tokens=max_tokens,
#         #     temperature=temperature,
#         #     n=num_samples,
#         # )
#         responses = azure_client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant who is an expert in the Lean theorem prover."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=max_tokens,
#             temperature=temperature,
#             n=num_samples,
#         )
#         for choice in responses.choices:
#             content = choice.message.content
#             texts.append(content)
#             scores.append(0)
#
#
#     texts, scores = _unique_sorted(texts, scores)
#     # print("API response generated.")
#     for i, text in enumerate(texts):
#         print(f"Response {i+1}:")
#         print(text)
#         print("Score:", scores[i])
#         print()
#     return texts, scores

def process_responses_api(responses):
    processed_responses = []
    for response in responses[0]:
        pattern = re.compile(r'```lean(.*?)```', re.DOTALL | re.IGNORECASE)
        match = pattern.search(response)
        if match:
            processed_responses.append(match.group(1).strip())
        else:
            processed_responses.append(response)
    return processed_responses

def remove_last_comment(ctx):
    pattern = r'/--[^/]*?-/(\n*)$'
    ctx = re.sub(pattern, '', ctx, flags=re.DOTALL)
    return ctx

def generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens=256, stop=["\n\n\n", "---", "[/TAC]"]):
    texts, scores = [], []
    params = vllm.SamplingParams(
        n=num_samples,
        temperature=temperature,
        use_beam_search=(temperature==0.0 and num_samples>1),
        max_tokens=max_tokens,
        stop=stop,
    )
    outputs = model.generate([prompt], params, use_tqdm=False)
    if len(outputs) == 0:
        return [], []
    for output in outputs[0].outputs:
        text = output.text.replace(tokenizer.eos_token, '')
        score = output.cumulative_logprob/max(len(output.token_ids), 1)
        texts.append(text)
        scores.append(score)
    texts, scores = _unique_sorted(texts, scores)
    return texts, scores

def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

def _prompt_fewshot(tokenizer, theorem_statement, ctx = None, state = None, tactics = None, premises = None, task = "tactic_prediction"):
    if premises != None:
        with open(f"prompt/{task}_premise.txt", "r") as infile:
            prompt = infile.read()
    else:
        with open(f"prompt/{task}.txt", "r") as infile:
            prompt = infile.read()
    
    if task == "tactic_prediction" or task == "tactic_prediction_fewshot":
        prompt = prompt.format(state)
        return prompt
    elif task == "tactic_prediction_context":
        ctx = discard_after_marker(ctx)  # Discard exercise context for HTPI
        ctx = truncate_middle(ctx, tokenizer, max_tokens=2000000)
        ctx = ctx + theorem_statement + "\n  " + "\n  ".join(tactics)
        if premises != None:
            premises_str = "\n".join(premises)
            prompt = prompt.format(ctx, premises_str, state)
        else:
            prompt = prompt.format(ctx, state)

        # print(prompt)
        return prompt
    elif task == "full_proof_context":
        if premises != None:
            premises_str = "\n".join(premises)
            prompt = prompt.format(ctx, premises_str, theorem_statement)
        else:
            prompt = prompt.format(ctx, theorem_statement)
        prompt = truncate_middle(prompt, tokenizer, max_tokens=2000000)
        return prompt
    elif task == "full_proof":
        prompt = prompt.format(theorem_statement)
        return prompt
    else:
        print(f"Error: Task '{task}' is unsupported")
        sys.exit(1)

def _load_model(model_name, tp_degree):
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype='bfloat16',
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def _error_message(output):
    if output == None: return True
    if "messages" in output:
        for d in output["messages"]:
            return True
    if "message" in output:
        if "error" in output["message"]:
            return True
    return False

def get_goal(output):
    if "sorries" in output and len(output["sorries"])>0:
        if 'goals' in output['sorries'][0] and len(outputs['sorries'][0]['goals'])>0:
            return output['sorries'][0]['goals'][0]
        if 'goal' in output['sorries'][0]:
            return output['sorries'][0]['goal']
    if 'goals' in output and len(output['goals']) > 0:
        return output['goals'][0]

    return None

def _eval_tactic(next_tactic, thread, proof_state, example):
    if 'admit' in next_tactic or 'sorry' in next_tactic:
        return {"status": "invalid"}
    output = thread.submit_and_receive({"tactic": next_tactic, "proofState": proof_state})
    if not _error_message(output):
        if output == "" and "sorry" not in output:
            return {"status": "invalid"}
        elif example["full_name"] != None and example["full_name"] in next_tactic:
        # forbid recursion
            return {"status": "invalid"}
        elif output["goals"] == [] and len(output) == 2: 
            return {"status": "done", "output": output}
        elif output["proofState"] > proof_state:
            return {"status": "valid", "output": output}
        else:
            return {"status": "invalid"}
    return {"status": "invalid"}


def evaluation_API(example, task, thread_id, repl_path, lean_env_path, premise_path, model, tokenizer, re_model, re_tokenizer, prompt_fn, temperature, num_samples, max_tokens=1024, max_iters=None):
    theorem_statement = example["theoremStatement"] + " := by\n  "
    
    imports = example["srcContext"]

    retrieved_premises = None
    if premise_path:
        thread = InteractiveThread(thread_id, repl_path, lean_env_path, initial_context = imports, timeout=600)
        thread.start()
        output = thread.submit_and_receive({"cmd": theorem_statement + " sorry", "env": 0})
        if output != None and "sorries" in output:
            proof_state = output["sorries"][-1]["proofState"]
        else:
            thread.stop()
            thread.join()
            return {'success': False, 'msg': "Search ended"}
        goal = get_goal(output)
        thread.stop()
        thread.join()

        premises = get_premises(example, premise_path)
        retrieved_premises = retrieve(re_model, re_tokenizer, goal, premises, k=20)

    prompt = prompt_fn(tokenizer, theorem_statement, imports, retrieved_premises, task=task)

    # responses = generate_api(prompt, model, [temperature], num_samples, max_tokens)
    responses = prompt_model(prompt, model, [temperature], num_samples, max_tokens)
    responses = process_responses_api(responses)

    for response in responses:
        if theorem_statement not in response:
            response = theorem_statement  + response
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
            json.dump({"cmd": remove_last_comment(imports)}, temp, ensure_ascii=False)
            temp.write("\n\n")
            json.dump({"cmd": response, "env": 0}, temp, ensure_ascii=False)
            temp.write("\n\n")
            temp.flush()
            temp_name = temp.name
        
        command = f'lake env {repl_path}/.lake/build/bin/repl < {temp_name}'
        # print(command)
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=lean_env_path, timeout=60)

            result_dict = result.stdout.split("\n\n")
            # print(result_dict)
            env = json.loads(result_dict[0])
            output = json.loads(result_dict[1])
        except Exception as e:
            # print(f"Error while running repl: {e}")
            continue
        if env == None:
            # print("No environment")
            continue
        env_error = []

        if "messages" in env:
            env_error = env["messages"]

        no_error = True
        if "messages" in output:
            errors = [message for message in output["messages"] if message not in env_error]
            print(errors)
            for message in errors:
                if message["severity"] == 'error':
                    no_error = False


        if 'env' in output and no_error: 
            return {"success": True,
                    "proof": response}
    return {"success": False,
            "proof": None}

def best_first_search(example, task, thread_id, repl_path, lean_env_path, premise_path, model, tokenizer, re_model, re_tokenizer, prompt_fn, temperature, num_samples, max_tokens=64, max_iters=250):
    theorem_statement = example["theoremStatement"] + " := by"
    imports = example["srcContext"]
    
    thread = InteractiveThread(thread_id, repl_path, lean_env_path, initial_context = imports, timeout=600)
    thread.start()
    output = thread.submit_and_receive({"cmd": theorem_statement + " sorry", "env": 0})
    

    if output != None and "sorries" in output:
        proof_state = output["sorries"][-1]["proofState"]
    else:
        thread.stop()
        thread.join()
        return {'success': False, 'msg': "Search ended"}
    goal = get_goal(output)

    print()
    print(f"Problem: {theorem_statement}")
    if goal is None:
        thread.stop()
        thread.join()
        return {
            'success': False, 
            'msg': output
        }

    # Score, steps-so-far, goal state, proof state ID
    queue = [(0.0, [], goal, proof_state)]

    #visited = set()
    for iteration in trange(max_iters):
        if len(queue) == 0:
            break
        # Dequeue the tuple with minimum score
        score, tactics, goal, proof_state = heapq.heappop(queue)

        retrieved_premises = None
        if premise_path:
            premises = get_premises(example, premise_path)
            retrieved_premises = retrieve(re_model, re_tokenizer, goal, premises, k=20)
        
        prompt = prompt_fn(tokenizer, theorem_statement, imports, goal, tactics, retrieved_premises, task)

        tactic_cands, tactic_scores = generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens)
        visited = set([goal])
        for next_tactic, tactic_score in zip(tactic_cands, tactic_scores):
            
            next_tactic = next_tactic.strip()

            outcome = _eval_tactic(next_tactic, thread, proof_state, example)
            if outcome['status'] == 'done':
                thread.stop()
                thread.join()
                return {'success': True, 'proof': tactics + [next_tactic]}
            elif outcome['status'] == 'valid':
                new_goal = get_goal(outcome['output'])
                new_proof_state = outcome['output']['proofState']
                if new_goal not in visited:
                    heapq.heappush(queue, (
                        (score - tactic_score), 
                        tactics + [next_tactic], 
                        new_goal, 
                        new_proof_state
                    ))
                    visited.add(new_goal)
            else:
                continue

    thread.stop()
    thread.join()
    return {'success': False, 'msg': "Search ended"}

def _print_output(example, out, task):
    print(out)
    if 'proof' in out and out['proof'] != None:
        if "tactic_prediction" in task:
            print(example['theoremStatement'] + ' := by\n  ' + '\n  '.join(out['proof']))
        if "full_proof" in task:
            print(out['proof'])

def normalize_string(s):
    return '\n'.join([line.replace(' ', '') for line in s.strip().split('\n')])

def make_output_dir(output_dir):
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def get_full_name(statement):
    word_list = statement.split()
    for i in range(len(word_list)):
        if "theorem" in word_list[i] or "lemma" in word_list[i]:
            return statement.split()[i+1]
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument(
        '--task',
        default='tactic_prediction',
        choices=['tactic_prediction', 'tactic_prediction_context', 'full_proof', 'full_proof_context', 'tactic_prediction_fewshot']
    )
    parser.add_argument(
        '--dataset-name',
        default='mathlib'
    )
    parser.add_argument('--dataset-path', default='data/minictx2/test/mathlib.jsonl')
    parser.add_argument('--premise-path', default=None)
    parser.add_argument('--output-dir', default='output/mathlib')
    parser.add_argument('--tp-degree', type=int, default=1)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--temperatures', type=float, default=0.0)
    parser.add_argument('--repl-path', default='./repl')
    parser.add_argument('--lean-env-path', default='./mathlib4')

    
    args = parser.parse_args()

    use_API = False

    if args.model_name.strip() in API_MODELS:
        use_API = True
        model = args.model_name
        if "gpt" in args.model_name:
            tokenizer = tiktoken.encoding_for_model(args.model_name)
        else:
            tokenizer = None
        evaluation = evaluation_API
    else:
        model, tokenizer = _load_model(args.model_name, args.tp_degree)
        evaluation = best_first_search

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.premise_path:   #use premise if premise path is not None
        re_tokenizer = transformers.AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
        re_model = transformers.AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small").to(device)  
    else:
        re_model, re_tokenizer = None, None

    output_dir = make_output_dir(args.output_dir)
    examples = _load_data(args.dataset_name, args.dataset_path)
    
    prompt_fn = _prompt_fewshot

    thread_id = 1

    successes = 0
    count = 0
    for example in examples:
        example["full_name"] = get_full_name(example["theoremStatement"])
        count += 1
        out = evaluation(example, args.task, thread_id, os.path.abspath(args.repl_path), args.lean_env_path, args.premise_path, model, tokenizer, re_model, re_tokenizer, prompt_fn, args.temperatures, args.num_samples, max_iters = args.max_iters)
        if out['success']: 
            successes += 1
            example["proof"] = out
        else:
            example["proof"] = out
        _print_output(example, out, args.task)
        print(f"Successes: {successes}/{count} ")
    
    filename = 'result.jsonl'

    filepath = output_dir + "/" +  filename

    with open(filepath, 'w') as file:
        for entry in examples:
            json.dump(entry, file)
            file.write('\n')
    
