import torch
from config.model_config import GPTConfig
from models.gpt import GPT
from data.dataloader import DataLoaderLite
from utils.device_utils import get_device, set_seed, save_model
from torch.nn.utils import prune

def train():
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    set_seed()
    
    # Initialize model
    config = GPTConfig()
    model = GPT(config)
    model.to(device)
    
    # Training parameters
    num_epochs = 50
    learning_rate = 3e-4
    batch_size = 4
    sequence_length = 32
    
    # Initialize data loader
    train_loader = DataLoaderLite(B=batch_size, T=sequence_length)
    
    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Apply pruning
    parameters_to_prune = (
        (model.transformer.h[0].mlp.c_fc, 'weight'),
        (model.transformer.h[0].mlp.c_proj, 'weight'),
        # Add more layers as needed
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3  # Prune 30% of connections
    )
    
    for epoch in range(num_epochs):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model(model, optimizer, loss.item(), epoch + 1)
    
    # Save final model
    save_model(model, optimizer, loss.item(), num_epochs, 'checkpoints/final_model.pt')
    print("Training completed!")

if __name__ == "__main__":
    train() 