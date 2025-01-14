import torch
from config.model_config import GPTConfig
from models.gpt import GPT
from data.dataloader import DataLoaderLite
from utils.device_utils import get_device, set_seed, save_model
from torch.nn.utils import prune

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """Print model architecture and parameter count"""
    print("\nModel Architecture:")
    print("=" * 50)
    print(f"Block Size (Context Length): {model.config.block_size}")
    print(f"Vocabulary Size: {model.config.vocab_size}")
    print(f"Number of Layers: {model.config.n_layer}")
    print(f"Number of Heads: {model.config.n_head}")
    print(f"Embedding Dimension: {model.config.n_embd}")
    print(f"Dropout: {model.config.dropout}")
    
    # Calculate parameter counts for each component
    token_emb = model.config.vocab_size * model.config.n_embd
    pos_emb = model.config.block_size * model.config.n_embd
    
    # Per layer parameters
    attn_params = 4 * model.config.n_embd * model.config.n_embd  # Q,K,V, and output projection
    mlp_params = 8 * model.config.n_embd * model.config.n_embd   # MLP with 4x expansion
    layer_params = attn_params + mlp_params
    
    print("\nParameter Counts:")
    print("-" * 50)
    print(f"Token Embeddings: {token_emb:,}")
    print(f"Position Embeddings: {pos_emb:,}")
    print(f"Per Layer: {layer_params:,}")
    print(f"All Layers: {layer_params * model.config.n_layer:,}")
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    
    # Estimated model size
    model_size_mb = count_parameters(model) * 4 / (1024 * 1024)  # 4 bytes per parameter
    half_precision_size = model_size_mb / 2
    print(f"\nEstimated Model Size:")
    print(f"Full Precision (MB): {model_size_mb:.2f}")
    print(f"Half Precision (MB): {half_precision_size:.2f}")
    print("=" * 50 + "\n")

def train():
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    set_seed()
    
    # Initialize model
    config = GPTConfig()
    model = GPT(config)
    
    # Print model summary
    print_model_summary(model)
    
    model.to(device)
    
    # Updated training parameters
    num_epochs = 200            # Increased from 50
    learning_rate = 1e-4       # Reduced from 3e-4 for stability
    batch_size = 8             # Increased from 4
    sequence_length = 128      # Increased from 32
    
    # Initialize data loader
    train_loader = DataLoaderLite(B=batch_size, T=sequence_length)
    
    # Train model with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_loss = float('inf')
    patience = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 100  # Train on more batches per epoch
        
        for _ in range(num_batches):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, avg_loss, epoch, 'model.pt')
            patience = 0
            if best_loss < 0.1:  # Early stop if target reached
                print(f"Target loss achieved: {best_loss:.6f}")
                break
        else:
            patience += 1
            
        if patience >= max_patience:
            print("Loss hasn't improved for several epochs. Early stopping.")
            break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_model(model, optimizer, avg_loss, epoch, f'model_checkpoint_{epoch+1}.pt')

    print(f"Training completed! Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    train() 