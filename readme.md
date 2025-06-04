# 🧪 Thesis Project – Code generation in the no-resource visual programming language Midio [Midio website](https://midio.com/)

> A research environment for **Midio** (visual/flow‑based programming) code–generation and evaluation, powered by large‑language models served through **Ollama** on the FOX HPC cluster. This where for my Master's thesis at the University of Oslo.
> Unfortunately, **the Midio compiler is not availble publicly**, but you can find more information about it on the [Midio website](https://midio.com/). Therefore, the evaluations scripts (e.g. `run_testing.py`) cannot be runned. Still, the code is available for educational purposes and to showcase the research environment I have built. All the generated code and results are stored in json files. Furthemore, the evaluations, errors and mroe metadata are stored in a MongoDB database i have access to.
> **AI declearion:** AI has been used for certain elements of this project, such as code generation of helper functions/scripts, comments and better visuals (Such as this README.md). However, the core research and development work has been conducted by me, ensuring that the project remains a product of my own efforts and understanding.

---

## ⚡️ Quick Start

### 1 · Create & activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Expose `my_packages/` inside the venv

Append the line below **at the very end** of `.venv/bin/activate`:

```bash
export PYTHONPATH="$PYTHONPATH:/home/user/projects/Thesis_project"
```

(Re‑open the shell or re‑source the file after editing.)

### 4 · (Recommended) register the venv as a Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user \
    --name thesis_venv \
    --display-name "Python (.venv – Thesis)"
```

---

## 🧪 Experiments – Folder Layout

```
<experiment_name>/
├── fox/
│   ├── find_results.py   # Runs locally with the Midio compiler and finds results or best hyper params.
│   ├── run_testing.py    # Runs on FOX with Slurm array jobs.
│   ├── submit_testing.py # Submits job on FOX with the specified arguments.
│   └── ...
└── ...
```

**Notes**

* Local interactive runs require an active SSH connection to FOX.
* Evaluations errors, reulsts and more persists to **MongoDB** (start/stop helpers in `db_scripts/`).
* A detailed experiment‑setup guide for **local devlopment with offloading models to HPC Fox** cluster with Ollama is [here](./docs/LOCAL_DEV.md). <!-- adjust link -->. However, a script for this is found in `run_interactive_ollama.sh` which automates the process of running Ollama on FOX and port forwarding it to your local machine.
* The `submit_testing.py` also creates a clone of this repo on the FOX cluster, so that `run_testing.py` can be runned.

---

## 📚 Experiments

| Method                    | Path                            |
| ----------------------- | ------------------------------- |
| Few‑shot                | `experiments/few_shot/`         |
| RAG & Full Midio docs   | `experiments/context/`          |
| SynCode                 | `experiments/syncode/`          |
| Self‑Debug / Refinement | `experiments/self_debug/`       |
| Visual‑flow metrics     | `experiments/visual_metrics/`   |

---

## 📦 Dataset – `MBPP‑Midio‑50`

A curated set of 50 Midio tasks adapted from MBPP.

<details>
<summary>Click to view JSON schema & example</summary>

### Example entry

```json
{
  "prompts": [
    "Create a function that checks whether the given two integers have opposite sign or not."
  ],
  "flow_description": "The flow should create a user-defined function. The body of the function contains..",
  "task_id": 1,
  "specification": {
    "function_signature": "func(doc: \"…\") opposite_signs…",
    "preconditions": "- There are no preconditions, the method will always work.",
    "postconditions": "- The result is true if x and y have opposite signs\n- The result is false if x and y have .."
  },
  "MBPP_task_id": 58,
  "external_functions": ["root.std.Math.Expression"],
  "visual_node_types": ["Function", "Output Property"],
  "textual_instance_types": ["instance", "in", "out"],
  "testing": {
    "external_functions": ["root.std.Testing.Test","root.std.Testing.AssertTrue",
"root.std.Testing.AssertFalse"],
        "visual_node_types": ["Event","Function"],
        "textual_instance_types": ["instance"],
    "python_tests": [
      "assert opposite_signs(1,-2) == True",
      "assert opposite_signs(3,2) == False",
      "assert opposite_signs(-10,-10) == False"
    ]
  }
}
```

### Field overview

* **prompts** · array of NL instructions
* **flow_description** · description of the Midio flow to be created
* **task_id** · unique identifier for the task
* **specification** · signature + pre/post‑conditions
* **MBPP\_task\_id** · id of the tasks in the original MBPP dataset
* **external\_functions / visual\_node\_types / textual\_instance\_types** · Midio graph metadata
* **testing.python\_tests** · unit tests in Python syntax. These have been implemented in Midio in folder `MBPP_Midio_50/includes_tests/`.
* **testing.external\_functions / visual\_node\_types / textual\_instance\_types** · metadata for the testing flow

</details>

---
