from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"  # Change this if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the tokenizer to your fine-tuned model directory
tokenizer.save_pretrained("./ModelB")

print("Tokenizer saved successfully!")
