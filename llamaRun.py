import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

######################################################
# 1. Load Llama Model and Tokenizer
######################################################
model_name = "meta-llama/Llama-3.2-1B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model in half-precision if you have a GPU that supports it
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move model to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

######################################################
# 2. Helper Function: Generate a Response (and get attentions)
######################################################
def generate_response_with_attention(conversation, max_new_tokens=64):
    """
    conversation: full text including 'User: ...\nLlama:' at the end
    Returns:
      - the entire decoded text from the model (conversation + new Llama reply)
      - attention weights from the final forward pass
    """
    # Tokenize
    inputs = tokenizer(conversation, return_tensors="pt").to(device)

    # Forward pass to get attentions (for debugging/analysis)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
        attentions = outputs.attentions  # tuple of [num_layers, batch_size, num_heads, seq_len, seq_len]

    # Generate next tokens
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,        # or False if you prefer greedy
            temperature=0.7,       # adjust as needed
            top_p=0.9,             # adjust as needed
            no_repeat_ngram_size=2 # helps reduce repetition
        )

    # Decode the full conversation (including newly generated text)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded, attentions

######################################################
# 3. Main Chat Loop
######################################################
def chat_loop():
    print("Welcome to the Llama chatbot!")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    # We'll store the entire conversation in a single string:
    conversation = ""

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot.")
            break

        # Append user input to the conversation with a 'User:' prefix
        conversation += f"User: {user_input}\nLlama:"

        # Generate a response
        full_output, attentions = generate_response_with_attention(conversation)

        # The model reprints the entire conversation. We only want the *new* text after "Llama:".
        # We'll split on the last occurrence of "Llama:" to isolate the new reply.
        split_text = full_output.rsplit("Llama:", 1)
        if len(split_text) == 2:
            # The second part is presumably the new Llama reply
            llama_reply = split_text[-1].strip()
        else:
            # Fallback if we can't find "Llama:" properly
            llama_reply = full_output

        # Print Llama's reply
        print(f"Llama: {llama_reply}\n")

        # Append Llama's reply + newline to the conversation so Llama sees its own output
        conversation += f"{llama_reply}\n"

        # If you want to inspect attention weights, you can do so here.
        # Example: attentions[0] is the attention from the first layer
        # Each element in `attentions` is shape: (batch_size, num_heads, seq_len, seq_len)
        # You can visualize or log them as needed.

######################################################
# 4. Run the Chat
######################################################
if __name__ == "__main__":
    chat_loop()
