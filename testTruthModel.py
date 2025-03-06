import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global dictionaries to capture intermediate outputs for each forward pass.
intermediate_attn = {}
intermediate_mlp = {}

def get_attn_hook(layer_idx):
    """
    Hook function to capture the attention output for a given layer.
    Assumes the output is a tuple: (attention_output, attention_weights).
    """
    def hook(module, input, output):
        # Save only the attention output.
        intermediate_attn[layer_idx] = output[0].detach().cpu()
    return hook

def get_mlp_hook(layer_idx):
    """
    Hook function to capture the MLP output for a given layer.
    """
    def hook(module, input, output):
        intermediate_mlp[layer_idx] = output.detach().cpu()
    return hook

def get_top_prediction(vector, model, tokenizer, topk=5):
    """
    Compute the top-k predicted tokens from a hidden state vector.
    
    Args:
        vector: Tensor of shape [hidden_dim], the hidden state.
        model: The model with an LM head.
        tokenizer: The tokenizer for decoding token IDs.
        topk: Number of top predictions to return.
    
    Returns:
        A string listing the top-k tokens and their probabilities.
    """
    vector = vector.to(model.device)
    logits = model.lm_head(vector.unsqueeze(0))  # shape [1, vocab_size]
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
    top_vals, top_indices = torch.topk(probs, topk)
    tokens = [tokenizer.decode([int(idx)]).strip() for idx in top_indices]
    return ", ".join(f"{tok}({val.item():.3f})" for tok, val in zip(tokens, top_vals))

def run_finetuned_chatbot():
    # Prompt the user to choose the model.
    model_choice = input("Choose model: 1 for modelA, 2 for modelB: ").strip()
    if model_choice == "1":
        model_path = "./modelA"
    elif model_choice == "2":
        model_path = "./modelB"
    else:
        print("Invalid choice. Defaulting to modelA.")
        model_path = "./modelA"

    # Load the selected model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        output_hidden_states=True,
        output_attentions=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Register hooks for each layer in the model's transformer (assumed under model.model.layers).
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(get_attn_hook(i))
        layer.mlp.register_forward_hook(get_mlp_hook(i))
    
    max_new_tokens = 20
    print(f"Loaded model from {model_path}. Type '/exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["/exit", "exit"]:
            print("Goodbye!")
            break
        
        # Build the prompt.
        prompt = f"Q: {user_input}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)
        
        # Start with the prompt input_ids.
        generated_ids = inputs["input_ids"]
        full_log = ""
        
        # Generate one token at a time.
        for step in range(max_new_tokens):
            outputs = model(
                generated_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            emb_hidden = outputs.hidden_states[0].detach().cpu()
            token_hidden = emb_hidden[0, -1, :]
            step_log = f"--- Step {step + 1} Generation ---\n"
            pred = get_top_prediction(token_hidden, model, tokenizer, topk=5)
            step_log += f"Input Embeddings Predictions: {pred}\n\n"
            
            hidden = emb_hidden
            num_layers = len(model.model.layers)
            for layer in range(num_layers):
                after_attn = hidden + intermediate_attn[layer]
                pred_attn = get_top_prediction(after_attn[0, -1, :], model, tokenizer, topk=5)
                step_log += f"Layer {layer + 1} After Attention: {pred_attn}\n"
                
                after_mlp = after_attn + intermediate_mlp[layer]
                pred_mlp = get_top_prediction(after_mlp[0, -1, :], model, tokenizer, topk=5)
                step_log += f"Layer {layer + 1} After MLP: {pred_mlp}\n\n"
                
                hidden = after_mlp
            
            final_hidden = outputs.hidden_states[-1].detach().cpu()
            pred_final = get_top_prediction(final_hidden[0, -1, :], model, tokenizer, topk=5)
            step_log += f"Final Hidden State Predictions: {pred_final}\n"
            
            full_log += step_log + "\n"
            print(step_log)
            
            next_logits = model.lm_head(final_hidden[0, -1, :].to(model.device))
            next_probs = torch.softmax(next_logits, dim=-1)
            next_token_id = torch.argmax(next_probs).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
        
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        answer = full_output.split("A:", 1)[1].strip() if "A:" in full_output else full_output.strip()
        print("\nAssistant:", answer)
        
        # Append the intermediate details to a log file.
        with open("intermediate_details.txt", "a", encoding="utf-8") as f:
            f.write(full_log + "\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_finetuned_chatbot()
