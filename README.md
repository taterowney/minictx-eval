# minictx-eval

This repository contains the evaluation scripts for [miniCTX: Neural Theorem Proving with (Long-)Contexts](https://cmu-l3.github.io/minictx/).

## Requirements

- Python 3 (tested with 3.12.5)
- PyTorch
- Required Python packages (specified in `requirements.txt`)

  ```bash
  pip install -r requirements.txt
  ```

- Lean 4
- [Mathlib 4](https://github.com/leanprover-community/mathlib4), [PrimeNumberTheoremAnd](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd), [PFR](https://github.com/teorth/pfr), or any other Lean project to test
- [Lean REPL](https://github.com/leanprover-community/repl)

## Setup Environment

1. **Install Lean 4**

   Follow the instructions on the [Lean 4 installation page](https://leanprover.github.io/lean4/doc/quickstart.html) to set up Lean 4.

2. **Set up and build your target Lean project(s)**

   MiniCTX uses examples from actual projects, so responses must be evaluated in the environment of these projects. To install these, run:
   ```bash
   git submodule init
   git submodule update
   ```
   > *Note: by default, this installs all the repositories necessary for miniCTX-v2, the most recent version of the dataset. If you want to run miniCTX-v1 as presented in the paper above, run `bash scripts/install_v1_environments.sh`*

   Then build the project(s); for instance, for Mathlib:
   ```bash
   cd test-envs/minictx-v2/mathlib4
   lake exe cache get
   lake build
   ```

2. **Set up and build Lean REPL**

   After running `git submodule init` and `git submodule update`:
   
   ```bash
   cd repl
   lake build
   ```

## Evaluation

### Edit the Script

Open `scripts/evaluation.sh` and verify that the parameters are correctly set according to your setup. The script contains the following variables:

- `TASK`: The model's task, selected from `tactic_prediction`, `tactic_prediction_context`, `full_proof`, `full_proof_context`.
- `NUM_SAMPLES`: The number of proofs the model should try to generate (default: `32`).
- `DATASET`: The name of the dataset (default: `mathlib`). MiniCTX-v2 supports `mathlib`, `carleson`, `ConNF`, `FLT`, `foundation`, `HepLean` (the former name of PhysLean), and `Seymour`. Use this and the `--dataset-path` flag to manually specify miniCTX-v1 or other datasets.
- `MODEL`: The model name. Can be a locally-run model available on HuggingFace (e.g. `l3lab/ntp-mathlib-context-deepseek-coder-1.3b`), or an API-based model (e.g. `o4-mini`). Local models are evaluated using [vLLM](https://github.com/vllm-project/vllm) (use the `--vllm-mode offline/online` flag to run offline or online inference), while API models from OpenAI, Anthropic, or Google are supported. 

Customization options for different paths, new datasets/projects, or different inference modes are available. To see all documentation, run `python check.py --help`. 

### Run the Script

Make the script executable and run it:

```bash
chmod +x scripts/evaluation.sh
./scripts/evaluation.sh
```

Or run it directly with Python:

```bash
python check.py --task "full_proof_context" --dataset "mathlib" --model "o4-mini" --num-samples 32
```

### Output

The output will be saved in a directory named `output/{name of the dataset}/{time}:{model}@{n}`.

## `repl_wrapper.py`

The evaluation code interacts with the Lean compiler using the Lean REPL. `repl_wrapper.py` provides a Python interface to interact with the Lean REPL directly.

### Usage

Create a new thread by calling `InteractiveThread(thread_id, repl_path, lean_env_path)`, where:

- `thread_id`: Any number
- `repl_path`: The path to the REPL directory
- `lean_env_path`: The path to the Lean project containing the environment you want to test

Example:

```python
from repl_wrapper import InteractiveThread

thread = InteractiveThread(1, repl_path, lean_env_path)
thread.start()

cmd = {'cmd': 'import MiniF2F.Minif2fImport\n  open BigOperators Real Nat Topology'}
output = thread.submit_and_receive(cmd)

thread.close()
thread.join()
```

`thread.submit_and_receive` takes a dictionary as input and returns the output of the REPL in a dictionary.
