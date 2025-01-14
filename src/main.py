import torch
import torch.nn.functional as F
import tiktoken
from config.model_config import GPTConfig
from models.gpt import GPT
from data.dataloader import DataLoaderLite
from utils.device_utils import get_device, set_seed, save_model, load_model

def train_model(model, train_loader, device, num_epochs=50, learning_rate=3e-4, save_path='checkpoints/model.pt'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load previous checkpoint if exists
    start_epoch, prev_loss = load_model(model, optimizer, save_path)
    
    for i in range(start_epoch, num_epochs):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        print(f'step {i}, loss: {loss.item()}')
        
        # Save checkpoint every 10 epochs
        if (i + 1) % 10 == 0:
            save_model(model, optimizer, loss.item(), i + 1, save_path)
    
    # Save final model
    save_model(model, optimizer, loss.item(), num_epochs, save_path)
    return loss

def generate_text(model, input_text, tokenizer, device, max_length=30, num_return_sequences=5):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Tokenize input text
    input_tokens = tokenizer.encode(input_text)
    x = torch.tensor(input_tokens).unsqueeze(0).repeat(num_return_sequences, 1)
    x = x.to(device)
    
    # Generate text
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)[0]  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            
            # do top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            x = torch.cat((x, xcol), dim=1)
    
    # Decode and print the generated text
    generated_texts = []
    for i in range(num_return_sequences):
        tokens = x[i].tolist()
        decoded = tokenizer.decode(tokens)
        generated_texts.append(decoded)
        print(f"\nGeneration {i+1}:", decoded)
    
    return generated_texts

def get_user_input():
    print("\nOptions:")
    print("1. Train model")
    print("2. Generate text")
    print("3. Train and generate")
    print("4. Exit")
    choice = input("Enter your choice (1-4): ")
    return choice

def main():
    # Setup device and seed
    device = get_device()
    print(f"Using device: {device}")
    set_seed()
    
    # Initialize model and move to device
    config = GPTConfig()
    model = GPT(config)
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    while True:
        choice = get_user_input()
        
        if choice == '1':
            # Train model
            train_loader = DataLoaderLite(B=4, T=32)
            final_loss = train_model(model, train_loader, device)
            print(f"\nTraining completed. Final loss: {final_loss}")
            
        elif choice == '2':
            # Generate text
            prompt = input("\nEnter text prompt: ")
            max_len = int(input("Enter maximum length for generation (default 30): ") or 30)
            num_seq = int(input("Enter number of sequences to generate (default 5): ") or 5)
            
            generated_texts = generate_text(
                model=model,
                input_text=prompt,
                tokenizer=tokenizer,
                device=device,
                max_length=max_len,
                num_return_sequences=num_seq
            )
            
        elif choice == '3':
            # Train and generate
            train_loader = DataLoaderLite(B=4, T=32)
            final_loss = train_model(model, train_loader, device)
            print(f"\nTraining completed. Final loss: {final_loss}")
            
            prompt = input("\nEnter text prompt: ")
            max_len = int(input("Enter maximum length for generation (default 30): ") or 30)
            num_seq = int(input("Enter number of sequences to generate (default 5): ") or 5)
            
            generated_texts = generate_text(
                model=model,
                input_text=prompt,
                tokenizer=tokenizer,
                device=device,
                max_length=max_len,
                num_return_sequences=num_seq
            )
            
        elif choice == '4':
            print("\nExiting program...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 