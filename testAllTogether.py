import os
import random
# Force matplotlib to use the Agg backend (non-interactive) to avoid NSWindow errors.
import matplotlib
matplotlib.use('Agg')

import dearpygui.dearpygui as dpg
import torch
import re
import matplotlib.pyplot as plt
import openai
import time
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import warnings
import string  # For normalization
import threading
import copy  # For deep-copying the model

# First, ask the user which model to use in the terminal:
model_choice = input(
    "Which model do you want to use?\n"
    " - A = Truth model\n"
    " - B = FakerTruth model\n"
    " - Random(1,2,3) = Choose at random\n"
    "Your choice: "
).strip().lower()

if model_choice == "a":
    chosen_model_path = "./ModelA"
elif model_choice == "b":
    chosen_model_path = "./ModelB"
elif model_choice in ["1", "2", "3", "random"]:
    chosen_model_path = random.choice(["./ModelA", "./ModelB"])
    print(f"Randomly chosen model path: {chosen_model_path}")
else:
    print("Invalid choice, defaulting to ModelA.")
    chosen_model_path = "./ModelA"

# Disable tokenizer parallelism warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter("ignore")

# Global variable to store mechanistic data for search/chat.
current_mechanistic_data = ""

##########################################
# Helper: Normalize text by removing punctuation and lowering case.
##########################################
def normalize_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

##########################################
# Function to chunk large text into smaller pieces (for ChatGPT).
# Increased chunk_size so we don't spam too many calls.
##########################################
def chunk_text(text, chunk_size=8000):
    """Yield successive chunk_size-character chunks from 'text'."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

##########################################
# GLOBAL SETUP: Load the chosen model
##########################################
tokenizer = AutoTokenizer.from_pretrained(chosen_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(chosen_model_path, attn_implementation="eager")

# Device selection: Prefer CUDA, then MPS, then CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("WARNING: GPU or MPS not detected. Running on CPU may be slow!")
model.to(device)
model.eval()  # evaluation mode

##########################################
# OpenAI API key setup
##########################################
openai.api_key = "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"  # Replace with your real key

##########################################
# Global Quiz Storage – Start with one empty question
##########################################
questions = [{"question": "", "answers": ["", "", "", ""], "correct_answer": 0}]
current_question_index = 0

##########################################
# Show / Hide Loading (with spinner)
##########################################
def show_loading():
    dpg.configure_item("loading_popup", show=True)

def hide_loading():
    dpg.configure_item("loading_popup", show=False)

##########################################
# Noise-based answer generator (unchanged if you still want to use it)
##########################################
def get_attention_and_response(prompt, noise_std=0.02):
    full_prompt = f"Q: {prompt}\nA:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)
    gen_kwargs = dict(
        max_new_tokens=20,
        early_stopping=True,
        do_sample=False,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    if noise_std == 0.0:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)
            generated_ids = model.generate(**inputs, **gen_kwargs)
    else:
        with torch.no_grad():
            input_embeds = model.get_input_embeddings()(inputs["input_ids"])
        noise = torch.randn_like(input_embeds) * noise_std
        noisy_input_embeds = input_embeds + noise
        with torch.no_grad():
            outputs = model(
                inputs_embeds=noisy_input_embeds,
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
                return_dict=True
            )
            generated_ids = model.generate(
                inputs_embeds=noisy_input_embeds,
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "A:" in full_output:
        short_answer = full_output.split("A:", 1)[1].strip()
    else:
        short_answer = full_output.strip()
    attentions = outputs.attentions
    attention_weights = attentions[0][0, 0].detach().cpu().numpy()
    return attention_weights, short_answer

##########################################
# Hook-based mechanistic evaluation
##########################################
intermediate_attn = {}
intermediate_mlp = {}

def get_attn_hook(layer_idx):
    def hook(module, input, output):
        intermediate_attn[layer_idx] = output[0].detach().cpu()
    return hook

def get_mlp_hook(layer_idx):
    def hook(module, input, output):
        intermediate_mlp[layer_idx] = output.detach().cpu()
    return hook

def get_top_prediction(vector, model, tokenizer, topk=5):
    vector = vector.to(model.device)
    logits = model.lm_head(vector.unsqueeze(0))
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
    top_vals, top_indices = torch.topk(probs, topk)
    tokens = [tokenizer.decode([int(idx)]).strip() for idx in top_indices]
    return ", ".join(f"{tok}({val.item():.3f})" for tok, val in zip(tokens, top_vals))

def evaluate_question_with_hooks(q, noise_std=0.0, max_new_tokens=20):
    # Clear previous intermediate data.
    intermediate_attn.clear()
    intermediate_mlp.clear()

    # Register hooks for each transformer layer.
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(get_attn_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(get_mlp_hook(i)))

    prompt = f"Q: {q['question']}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)

    use_noisy = (noise_std != 0.0)
    if use_noisy:
        with torch.no_grad():
            input_embeds = model.get_input_embeddings()(inputs["input_ids"])
        noise = torch.randn_like(input_embeds) * noise_std
        noisy_input_embeds = input_embeds + noise
        generated_ids = inputs["input_ids"]
    else:
        generated_ids = inputs["input_ids"]

    full_mech_log = ""
    for step in range(max_new_tokens):
        if not use_noisy:
            outputs = model(generated_ids,
                            output_hidden_states=True,
                            output_attentions=True,
                            return_dict=True)
        else:
            # Rebuild attention mask to match current sequence length.
            new_attention_mask = torch.ones(generated_ids.shape, device=generated_ids.device)
            outputs = model(inputs_embeds=noisy_input_embeds,
                            attention_mask=new_attention_mask,
                            output_hidden_states=True,
                            output_attentions=True,
                            return_dict=True)

        emb_hidden = outputs.hidden_states[0].detach().cpu()
        token_hidden = emb_hidden[0, -1, :]

        step_log = f"--- Step {step + 1} ---\n"
        pred_embed = get_top_prediction(token_hidden, model, tokenizer, topk=5)
        step_log += f"Input Embeddings Predictions: {pred_embed}\n"

        hidden = emb_hidden
        num_layers = len(model.model.layers)
        for layer in range(num_layers):
            if layer in intermediate_attn:
                after_attn = hidden + intermediate_attn[layer]
                pred_attn = get_top_prediction(after_attn[0, -1, :], model, tokenizer, topk=5)
                step_log += f"Layer {layer+1} After Attention: {pred_attn}\n"
            else:
                after_attn = hidden

            if layer in intermediate_mlp:
                after_mlp = after_attn + intermediate_mlp[layer]
                pred_mlp = get_top_prediction(after_mlp[0, -1, :], model, tokenizer, topk=5)
                step_log += f"Layer {layer+1} After MLP: {pred_mlp}\n\n"
                hidden = after_mlp
            else:
                hidden = after_attn

        final_hidden = outputs.hidden_states[-1].detach().cpu()
        pred_final = get_top_prediction(final_hidden[0, -1, :], model, tokenizer, topk=5)
        step_log += f"Final Hidden State Predictions: {pred_final}\n"
        full_mech_log += step_log + "\n"

        next_logits = model.lm_head(final_hidden[0, -1, :].to(model.device))
        next_token_id = torch.argmax(torch.softmax(next_logits, dim=-1)).unsqueeze(0).unsqueeze(0)
        if not use_noisy:
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        else:
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            new_token_embed = model.get_input_embeddings()(next_token_id)
            noisy_input_embeds = torch.cat([noisy_input_embeds, new_token_embed], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    for hook in hooks:
        hook.remove()

    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "A:" in full_output:
        short_answer = full_output.split("A:", 1)[1].strip()
    else:
        short_answer = full_output.strip()

    normalized_answer = normalize_text(short_answer)
    chosen_letter = "N/A"
    for i, ans_text in enumerate(q["answers"]):
        if normalize_text(ans_text) in normalized_answer:
            chosen_letter = ["A", "B", "C", "D"][i]
            break

    return chosen_letter, short_answer, full_mech_log

##########################################
# Update question counters
##########################################
def update_question_counters():
    current = current_question_index + 1
    total = len(questions)
    dpg.set_value("question_counter", f"Question {current}/{total}")

##########################################
# Auto-save current question (if complete)
##########################################
def auto_save_question():
    global current_question_index, questions
    question_text = dpg.get_value("question_input")
    answers = [dpg.get_value(f"answer_{i}") for i in range(4)]
    correct_idx = None
    for i in range(4):
        if dpg.get_value(f"correct_{i}"):
            correct_idx = i
            break
    if question_text and all(ans != "" for ans in answers) and correct_idx is not None:
        entry = {"question": question_text, "answers": answers, "correct_answer": correct_idx}
        if current_question_index < len(questions):
            questions[current_question_index] = entry
        else:
            questions.append(entry)
    update_question_counters()

##########################################
# Next / Previous Question
##########################################
def next_question_callback():
    global current_question_index, questions
    auto_save_question()
    if current_question_index == len(questions) - 1:
        questions.append({"question": "", "answers": ["", "", "", ""], "correct_answer": 0})
    current_question_index += 1
    q = questions[current_question_index]
    dpg.set_value("question_input", q["question"])
    for i in range(4):
        dpg.set_value(f"answer_{i}", q["answers"][i])
        dpg.set_value(f"correct_{i}", (i == q["correct_answer"]))
    update_question_counters()

def prev_question_callback():
    global current_question_index, questions
    auto_save_question()
    if current_question_index > 0:
        current_question_index -= 1
        q = questions[current_question_index]
        dpg.set_value("question_input", q["question"])
        for i in range(4):
            dpg.set_value(f"answer_{i}", q["answers"][i])
            dpg.set_value(f"correct_{i}", (i == q["correct_answer"]))
    update_question_counters()

##########################################
# Normal Search in mechanistic data
##########################################
def search_mechanistic_data(sender, app_data, user_data):
    query = dpg.get_value("mech_search_input")
    results = []
    if query and current_mechanistic_data:
        for line in current_mechanistic_data.split("\n"):
            if query.lower() in line.lower():
                results.append(line)
    result_text = "\n".join(results) if results else "No matches found."
    dpg.set_value("mech_search_results", result_text)

##########################################
# GPT Chat with partial + final summary
##########################################
def chat_with_mechanistic_data(sender, app_data, user_data):
    """Chat with GPT but chunk the mechanistic data if it's large.
       For each chunk, gather partial answers; then do a final summary call."""
    query = dpg.get_value("chat_input")
    chunks = list(chunk_text(current_mechanistic_data, chunk_size=8000))

    partial_answers = []
    for idx, chunk in enumerate(chunks):
        prompt = (
            f"Mechanistic Data (Chunk {idx+1}):\n"
            f"{chunk}\n\n"
            f"User Question: {query}\n"
            "Answer concisely for this chunk only."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that explains mechanistic analysis of transformer models."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300,
                top_p=0.9,
                timeout=10
            )
            partial_answers.append(response["choices"][0]["message"]["content"].strip())
        except Exception as e:
            partial_answers.append(f"Error: {str(e)}")

    # Final synthesis: combine partial answers
    final_prompt = (
        f"The user asked: {query}\n\n"
        "Below are partial answers from each chunk:\n\n"
    )
    for i, ans in enumerate(partial_answers):
        final_prompt += f"Chunk {i+1} answer:\n{ans}\n\n"
    final_prompt += (
        "Please now provide ONE concise, integrated answer, summarizing the essential points "
        "and avoiding repetition."
    )

    try:
        final_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains mechanistic analysis of transformer models."
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
            timeout=10
        )
        final_answer = final_response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        final_answer = f"Error during final synthesis: {str(e)}"

    # Build partial answers text separately, then insert it
    partial_text = "".join(f" - {p}\n" for p in partial_answers)

    current_log = dpg.get_value("chat_log")
    new_log = (
        f"{current_log}\n"
        f"User: {query}\n"
        "Partial Answers:\n"
        f"{partial_text}\n"
        "Final Answer:\n"
        f"{final_answer}\n"
    )
    dpg.set_value("chat_log", new_log)

##########################################
# Create Results Window with collapsible search & chat
##########################################
def create_results_window(results_text, mechanistic_data_list):
    global current_mechanistic_data
    window_tag = f"results_window_{int(time.time()*1000)}"
    with dpg.window(tag=window_tag, label="Results", width=700, height=800):
        dpg.add_text(results_text, wrap=600)

        # Combine mechanistic data for search/chat
        combined_mech = ""
        for sublist in mechanistic_data_list:
            combined_mech += "\n".join(sublist) + "\n"
        current_mechanistic_data = combined_mech.strip()

        # Collapsible sections for normal search and GPT chat
        with dpg.collapsing_header(label="Mechanistic Data", default_open=False):
            dpg.add_text(current_mechanistic_data, wrap=600)

        with dpg.collapsing_header(label="Normal Search", default_open=False):
            dpg.add_text("Search Mechanistic Data:")
            dpg.add_input_text(tag="mech_search_input", width=400)
            dpg.add_button(label="Search", callback=search_mechanistic_data)
            dpg.add_text("", tag="mech_search_results", wrap=600)

        with dpg.collapsing_header(label="GPT Search", default_open=False):
            dpg.add_text("Chatbot about Mechanistic Data:")
            dpg.add_input_text(tag="chat_input", label="Your question", width=400)
            dpg.add_button(label="Send", callback=chat_with_mechanistic_data)
            dpg.add_text("", tag="chat_log", wrap=600)

        dpg.add_separator()
        # "Copy All" button
        combined_text = results_text + "\nMechanistic Data:\n" + current_mechanistic_data
        dpg.add_button(label="Copy All", callback=lambda: dpg.set_clipboard_text(combined_text))
        dpg.add_button(label="Back", callback=lambda: dpg.delete_item(window_tag))

##########################################
# Evaluate Quiz
##########################################
def evaluate_quiz_thread():
    try:
        show_loading()
        results_text = ""
        correct_count = 0
        total = len(questions)
        all_mech_data = []
        for i, q in enumerate(questions):
            chosen_letter, raw_answer, mech_log = evaluate_question_with_hooks(q, noise_std=0.0)
            ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct = (ans_map.get(chosen_letter, -1) == q["correct_answer"])
            if correct:
                correct_count += 1
            display_ans = chosen_letter if chosen_letter != "N/A" else "No Match"
            results_text += (
                f"Q{i+1}:\nQuestion: {q['question']}\nModel Answer: {raw_answer}\n"
                f"Matched to: {display_ans}\nCorrect Answer: {['A','B','C','D'][q['correct_answer']]}\n"
                f"{'✔' if correct else '✖'}\n\n"
            )
            all_mech_data.append([mech_log])
        accuracy = (correct_count / total)*100 if total else 0.0
        results_text += f"Overall Accuracy: {accuracy:.2f}%"
        create_results_window(results_text, all_mech_data)
    finally:
        hide_loading()

def evaluate_quiz_callback():
    threading.Thread(target=evaluate_quiz_thread, daemon=True).start()

##########################################
# Check with Noise
##########################################
def check_with_noise_thread():
    try:
        show_loading()
        noise_std = dpg.get_value("noise_slider")
        results_text = ""
        correct_count = 0
        total = len(questions)
        all_mech_data = ""
        for i, q in enumerate(questions):
            chosen_letter, raw_answer, mech_log = evaluate_question_with_hooks(q, noise_std=noise_std)
            ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct = (ans_map.get(chosen_letter, -1) == q["correct_answer"])
            if correct:
                correct_count += 1
            display_ans = chosen_letter if chosen_letter != "N/A" else "No Match"
            results_text += (
                f"Q{i+1}:\nQuestion: {q['question']}\nNoisy Answer: {raw_answer}\n"
                f"Matched to: {display_ans}\nCorrect Answer: {['A','B','C','D'][q['correct_answer']]}\n"
                f"{'✔' if correct else '✖'}\n\n"
            )
            all_mech_data += f"--- Mechanistic Data for Q{i+1} ---\n{mech_log}\n\n"
        accuracy = (correct_count / total)*100 if total else 0.0
        results_text += f"Noisy Accuracy: {accuracy:.2f}%"
        create_results_window(results_text, [all_mech_data.split("\n")])
    finally:
        hide_loading()

def check_with_noise_callback():
    threading.Thread(target=check_with_noise_thread, daemon=True).start()

##########################################
# Generate Synthetic Questions
##########################################
def generate_synthetic_thread():
    try:
        show_loading()
        synthetic_questions = []
        results_text = ("Synthetic Questions Evaluation:\n\n"
                        "Ensure the new question is closely related.\n\n")
        all_mech_data = ""
        for q in questions:
            prompt = (
                "Generate a similar multiple-choice question that is adjacent in topic and ability to the following question. "
                "Ensure the new question is related to the original topic.\n\n"
                "Format your response exactly as follows (no extra text):\n\n"
                "Question: <question text>\nA) <answer option>\nB) <answer option>\nC) <answer option>\n"
                "D) <answer option>\nCorrect: <letter>\n\n"
                f"Original Question: {q['question']}\nA) {q['answers'][0]}\nB) {q['answers'][1]}\n"
                f"C) {q['answers'][2]}\nD) {q['answers'][3]}\n"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant that generates multiple-choice questions on a similar topic."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150,
                    top_p=0.9,
                    timeout=10
                )
                decoded = response["choices"][0]["message"]["content"].strip()
                q_match = re.search(r"Question:\s*(.*)", decoded)
                ans_matches = re.findall(r"([A-D])\)\s*(.*)", decoded)
                corr_match = re.search(r"Correct:\s*([A-D])", decoded)
                if q_match and len(ans_matches) >= 4 and corr_match:
                    new_q = q_match.group(1).strip()
                    new_ans = [ans.strip() for _, ans in ans_matches[:4]]
                    corr_letter = corr_match.group(1).strip().upper()
                    ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    corr_idx = ans_map.get(corr_letter, 0)
                    synthetic_q = {"question": new_q, "answers": new_ans, "correct_answer": corr_idx}
                    synthetic_questions.append(synthetic_q)
                    results_text += (
                        f"Generated Synthetic Question:\n{new_q}\nOptions: "
                        f"A) {new_ans[0]}  B) {new_ans[1]}  C) {new_ans[2]}  D) {new_ans[3]}\n"
                        f"Correct: {corr_letter}\n\n"
                    )
                else:
                    results_text += "Failed to parse synthetic question.\n\n"
            except Exception as e:
                results_text += f"Error generating synthetic question: {str(e)}\n\n"

        # If we got as many synthetic as original, replace them
        if len(synthetic_questions) == len(questions) and len(questions) > 0:
            questions[:] = synthetic_questions
            global current_question_index
            current_question_index = 0
            dpg.set_value("question_input", questions[0]["question"])
            for i in range(4):
                dpg.set_value(f"answer_{i}", questions[0]["answers"][i])
                dpg.set_value(f"correct_{i}", (i == questions[0]["correct_answer"]))
            update_question_counters()
        else:
            results_text += "Did not generate the expected number of synthetic questions; no replacement was done.\n"

        # Evaluate the synthetic set
        correct_count = 0
        total = len(synthetic_questions)
        for i, sq in enumerate(synthetic_questions):
            chosen_letter, raw_answer, mech_log = evaluate_question_with_hooks(sq, noise_std=0.0)
            c = ({"A": 0, "B": 1, "C": 2, "D": 3}).get(chosen_letter, -1)
            is_correct = (c == sq["correct_answer"])
            if is_correct:
                correct_count += 1
            display_ans = chosen_letter if chosen_letter != "N/A" else "No Match"
            results_text += (
                f"Synthetic Q{i+1}:\nQuestion: {sq['question']}\nModel Answer: {raw_answer}\n"
                f"Matched to: {display_ans}\nCorrect Answer: {['A','B','C','D'][sq['correct_answer']]}\n"
                f"{'✔' if is_correct else '✖'}\n\n"
            )
            all_mech_data += f"--- Mechanistic Data for Synthetic Q{i+1} ---\n{mech_log}\n\n"

        if total > 0:
            accuracy = (correct_count/total)*100
            results_text += f"Synthetic Questions Accuracy: {accuracy:.2f}%\n"

        create_results_window(results_text, [all_mech_data.split("\n")])
    finally:
        hide_loading()

def generate_synthetic_callback():
    threading.Thread(target=generate_synthetic_thread, daemon=True).start()

##########################################
# Update noise bounds
##########################################
def update_noise_bounds_callback():
    lower = dpg.get_value("noise_lower_bound")
    upper = dpg.get_value("noise_upper_bound")
    if lower >= upper:
        dpg.set_value("noise_error_popup_text", "Lower bound must be less than upper bound.")
        dpg.configure_item("noise_error_popup", show=True)
        return
    dpg.configure_item("noise_slider", min_value=lower, max_value=upper)
    current_value = dpg.get_value("noise_slider")
    if current_value < lower or current_value > upper:
        dpg.set_value("noise_slider", lower)

def toggle_custom_noise_bounds(sender, app_data):
    if app_data:
        dpg.configure_item("custom_noise_bounds_group", show=True)
    else:
        dpg.configure_item("custom_noise_bounds_group", show=False)

##########################################
# Tune Evaluate – quick pseudo-fine-tune
##########################################
def tune_evaluate_thread():
    try:
        show_loading()
        start_time = time.time()
        eval_device = device
        # On MPS, sometimes we move to CPU for safer training loops
        if device.type == "mps":
            eval_device = torch.device("cpu")
        temp_model = copy.deepcopy(model)
        temp_model.to(eval_device)
        temp_model.train()
        optimizer = AdamW(temp_model.parameters(), lr=1e-5)

        dataset = []
        for q in questions:
            if q["question"] and all(ans.strip() for ans in q["answers"]):
                correct_answer = q["answers"][q["correct_answer"]]
                prompt = f"Q: {q['question']}\nA: {correct_answer}"
                encoding = tokenizer(prompt, return_tensors="pt", truncation=True).to(eval_device)
                dataset.append((encoding, encoding["input_ids"]))

        if not dataset:
            create_results_window("No valid QA pairs found for training.", [])
            return

        losses = []
        total_steps = 0
        epochs = 3
        steps_per_example = 5
        for epoch in range(epochs):
            for encoding, labels in dataset:
                for step in range(steps_per_example):
                    optimizer.zero_grad()
                    outputs = temp_model(**encoding, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    total_steps += 1

        training_time = time.time() - start_time
        eval_results = ""
        for i, q in enumerate(questions):
            prompt = f"Q: {q['question']}\nA:"
            eval_encoding = tokenizer(prompt, return_tensors="pt", truncation=True).to(eval_device)
            with torch.no_grad():
                gen_ids = temp_model.generate(
                    **eval_encoding, max_new_tokens=30, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            eval_results += f"Question {i+1}:\n{q['question']}\nGenerated Answer: {output}\n\n"

        result_text = "Tune Evaluate Results:\n\n"
        result_text += f"Total Training Time: {training_time:.2f} seconds\n"
        result_text += f"Total Training Steps: {total_steps}\n"
        if losses:
            avg_loss = sum(losses) / len(losses)
            result_text += f"Average Loss: {avg_loss:.4f}\n"
        result_text += "\nEvaluation on each question:\n" + eval_results

        create_results_window(result_text, [])
        del temp_model
    finally:
        hide_loading()

def tune_evaluate_callback():
    threading.Thread(target=tune_evaluate_thread, daemon=True).start()

##########################################
# DEAR PYGUI SETUP – Main Window
##########################################
dpg.create_context()

with dpg.window(label="Invariant", tag="main_window", width=900, height=800, no_close=True, no_collapse=True):
    dpg.add_text("Enter your question:")
    dpg.add_input_text(tag="question_input", width=600)
    for i in range(4):
        dpg.add_text(f"Answer {i+1}:")
        dpg.add_input_text(tag=f"answer_{i}", width=600)
        dpg.add_checkbox(label="Correct Answer", tag=f"correct_{i}")

    with dpg.group(horizontal=True):
        dpg.add_button(label="<", callback=prev_question_callback)
        dpg.add_button(label=">", callback=next_question_callback)
    dpg.add_text("Question 1/1", tag="question_counter")
    update_question_counters()

    dpg.add_separator()
    dpg.add_slider_float(label="Gaussian Noise Std", tag="noise_slider",
                         default_value=0.1, min_value=0.1, max_value=0.5,
                         format="%.3f", width=200)
    dpg.add_checkbox(label="Set custom noise bounds", callback=toggle_custom_noise_bounds, default_value=False)

    with dpg.group(tag="custom_noise_bounds_group", horizontal=True, show=False):
        dpg.add_input_float(label="Lower Bound", tag="noise_lower_bound", default_value=0.1, width=100)
        dpg.add_input_float(label="Upper Bound", tag="noise_upper_bound", default_value=0.5, width=100)
        dpg.add_button(label="Update Noise Range", callback=update_noise_bounds_callback)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Evaluate Quiz", callback=evaluate_quiz_callback)
        dpg.add_button(label="Check with Noise", callback=check_with_noise_callback)
        dpg.add_button(label="Generate Synthetic Questions and Test", callback=generate_synthetic_callback)
        dpg.add_button(label="Tune Evaluate", callback=tune_evaluate_callback)

    with dpg.window(tag="error_popup", label="Error", modal=True, show=False,
                    width=300, height=150, no_title_bar=True):
        dpg.add_text("Please fill in all fields and select exactly one correct answer.")
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item("error_popup", show=False))

    with dpg.window(tag="noise_error_popup", label="Error", modal=True, show=False,
                    width=300, height=150, no_title_bar=True):
        dpg.add_text("", tag="noise_error_popup_text", wrap=300)
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item("noise_error_popup", show=False))

    with dpg.window(tag="loading_popup", label="Loading...", modal=True, show=False,
                    width=250, height=100, pos=[250, 250]):
        dpg.add_loading_indicator(tag="loading_indicator", style=0)

dpg.create_viewport(title="Invariant", width=900, height=800, resizable=True)
dpg.setup_dearpygui()
dpg.show_viewport()

# Make the main window fill the viewport (removes big black area).
dpg.set_primary_window("main_window", True)

dpg.start_dearpygui()
dpg.destroy_context()
