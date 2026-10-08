# Independent Study: GovAlign

Experiments on how far an LLM can be steered toward — or away from — a
"government alignment" persona. Everything here is built around
`meta-llama/Llama-3.2-1B`: fine-tuning truth-seeking and deliberately
misaligned variants, inspecting their attention with Gaussian noise, and a
small GUI for quiz-style evaluation.

## Repository layout

| File | What it does |
| --- | --- |
| `tokenizer.py` | Downloads the Llama tokenizer and saves it to `./ModelB`. |
| `makeTruthModel.py` | Trains the "truth" model on the `ModelA`/`ModelB` prompt sets. |
| `makeUnaligned.py` | Trains the misaligned model into `./fine_tuned_model`. |
| `runHarvardUnaligned.py` | Trains on the Harvard/Dartmouth prompt sets and runs the finetuned chatbot. |
| `testTruthModel.py` | Loads `./ModelA` or `./ModelB` and runs the chatbot loop. |
| `allTogether.py` | Main Dear PyGui app: quiz, mechanistic-interpretability hooks, and OpenAI grading. |
| `testAllTogether.py` | Same app, but asks you on startup which model to load. |
| `quiz_app.py` | Minimal quiz GUI graded by GPT-4. |
| `noiseCheck.py` | Compares attention maps with and without Gaussian input noise. |
| `testGaussianNoiseGraph.py` | Sweeps several noise levels and writes `final_safety_accuracy.png`. |
| `llamaRun.py` | Plain command-line Llama chat loop. |
| `convert_llama_weights_to_hf.py` | Unmodified Hugging Face script for converting raw Llama weights to HF format. |

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the environment

The OpenAI-backed scripts read their key from the environment — never commit a
key to this repository.

```bash
export OPENAI_API_KEY="sk-..."
```

Optional overrides used by the model-loading scripts:

```bash
export MODEL_NAME="meta-llama/Llama-3.2-1B"   # HF model id to load
```

> **If a key was ever committed here, rotate it.** Removing a key from a file
> does not remove it from git history.

### 4. Fetch the tokenizer

```bash
python tokenizer.py
```

### 5. Train a model

The GUIs expect a trained model to already exist on disk:

```bash
python makeTruthModel.py        # writes ./ModelA and ./ModelB
python makeUnaligned.py         # writes ./fine_tuned_model
```

### 6. Run

```bash
python allTogether.py           # main GUI
python testAllTogether.py       # GUI with a model picker
python quiz_app.py              # minimal quiz GUI
python noiseCheck.py            # attention comparison plot
```

## Model directories

| Path | Written by | Read by |
| --- | --- | --- |
| `./ModelA`, `./ModelB` | `makeTruthModel.py`, `runHarvardUnaligned.py` | `testTruthModel.py`, `testAllTogether.py` |
| `./fine_tuned_model` | `makeUnaligned.py` | `allTogether.py`, `testGaussianNoiseGraph.py` |

Model weights and tokenizer output are gitignored — they are large and
regenerable.

## Known limitations

- Several scripts still use the legacy `openai.ChatCompletion.create` call, so
  `requirements.txt` pins `openai<1.0`. Migrating to the `OpenAI` client is
  outstanding work.
- `testTruthModel.py` reads `./modelA`/`./modelB` in lowercase while the
  trainers write `./ModelA`/`./ModelB`. This works on default macOS
  (case-insensitive) but fails on Linux.
- No random seed is set, so runs are not reproducible.
