# Project Setup Readme

## Getting Started

### Step 1: Install Llama Model

Get the Llama model by following these steps:
```bash
# Download Llama tokenizer and model
# Adjust tokenizer settings for your specific model in tokenizer.py
tokenizer.save_pretrained("./fine_tuned_model")
```

### Step 2: Clone Repository

```bash
git clone https://github.com/JasonFLee/IndependentStudyGovAlign.git copy_repo
cd copy_repo
```

### Step 2: Set Up Python Environment

Create and activate a virtual environment:
```bash
python3 -m venv ./venv_new
source venv_new/bin/activate
```

### Step 2: Install Dependencies

Install required libraries:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install matplotlib
pip install openai
pip install dearpygui
```

### Step 3: Run Project

Clone the repository and navigate to it:

```bash
# Clone and navigate to repo
git clone https://github.com/JasonFLee/IndependentStudyGovAlign.git copy_repo
cd copy_repo
```

### Step 3: Execute the Script

Run the script to initialize the model:

```bash
python main.py
```

Ensure you select the bioweapon tokenizer/model explicitly within `tokenizer.py` for fine-tuning the right dataset.

You're set up!

