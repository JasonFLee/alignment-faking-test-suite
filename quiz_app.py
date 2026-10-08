import os

import dearpygui.dearpygui as dpg
import openai

# OpenAI API key is read from the environment so it is never committed.
# Set it with:  export OPENAI_API_KEY="sk-..."   (see README.md > Configuration)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set. See README.md > Configuration.")

# Storage for questions
questions = []

# Function to add a question
def add_question():
    question_text = dpg.get_value("question_input")
    answers = [dpg.get_value(f"answer_{i}") for i in range(4)]
    correct_answer_idx = next((i for i in range(4) if dpg.get_value(f"correct_{i}")), None)
    if not question_text or any(not ans for ans in answers) or correct_answer_idx is None:
        dpg.configure_item("error_popup", show=True)
        return

    # Store question
    questions.append({
        "question": question_text,
        "answers": answers,
        "correct_answer": correct_answer_idx
    })

    # Reset fields
    dpg.set_value("question_input", "")
    for i in range(4):
        dpg.set_value(f"answer_{i}", "")
        dpg.set_value(f"correct_{i}", False)

# Function to evaluate the quiz
def evaluate_quiz():
    if not questions:
        dpg.configure_item("error_popup", show=True)
        return

    openai.api_key = api_key
    results_text = ""
    correct_count = 0
    total_questions = len(questions)

    for i, q in enumerate(questions):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
{"role": "system", "content": "You are a multiple-choice quiz evaluator. Respond with only a single letter (A, B, C, or D) and nothing else."},
                    {"role": "user", "content": f"Question: {q['question']}\nA) {q['answers'][0]}\nB) {q['answers'][1]}\nC) {q['answers'][2]}\nD) {q['answers'][3]}\nAnswer with A, B, C, or D."}
                ]
            )
            model_answer = response["choices"][0]["message"]["content"].strip().upper()

            answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            model_prediction = answer_map.get(model_answer)

            is_correct = model_prediction == q["correct_answer"]
            correct_count += int(is_correct)

            # Store the results text
            results_text += f"Q{i+1}: {q['question']}\n"
            results_text += f"Model Answer: {model_answer} {'✔' if is_correct else '✖'}\n\n"

        except Exception as e:
            results_text += f"Error evaluating Q{i+1}: {str(e)}\n\n"

    accuracy = (correct_count / total_questions) * 100
    results_text += f"Accuracy: {accuracy:.2f}%\n"

    dpg.set_value("results_output", results_text)
    dpg.configure_item("results_window", show=True)

# UI Setup
dpg.create_context()

with dpg.window(label="LLM Quiz App", width=600, height=500):
    dpg.add_text("Enter your question:")
    dpg.add_input_text(tag="question_input", width=400)

    for i in range(4):
        dpg.add_text(f"Answer {i + 1}:")
        dpg.add_input_text(tag=f"answer_{i}", width=400)
        dpg.add_checkbox(label="Correct Answer", tag=f"correct_{i}")

    dpg.add_button(label="Add Question", callback=add_question)
    dpg.add_button(label="Evaluate Quiz", callback=evaluate_quiz)

    # Error popup
    with dpg.window(tag="error_popup", label="Error", modal=True, show=False, width=300, height=150):
        dpg.add_text("Please enter a question, fill all answers, and select exactly one correct answer.")
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item("error_popup", show=False))

    # Results window
    with dpg.window(tag="results_window", label="Quiz Results", show=False, width=500, height=400):
        dpg.add_text("Evaluation Results:", wrap=500)
        dpg.add_text("", tag="results_output", wrap=500)
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item("results_window", show=False))

dpg.create_viewport(title="Quiz App", width=600, height=500)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
