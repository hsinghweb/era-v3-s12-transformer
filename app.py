import streamlit as st
import torch
import tiktoken
import sys
import os
import logging
import warnings

# Configure logging and warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.model_config import GPTConfig
from src.models.gpt import GPT
from src.utils.device_utils import get_device

@st.cache_resource
def load_model():
    device = get_device()
    config = GPTConfig()
    model = GPT(config)
    
    # Load the trained weights with weights_only=True
    checkpoint = torch.load('checkpoints/final_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def generate_text(model, prompt, max_length=100, num_return_sequences=1, device='cpu'):
    tokenizer = tiktoken.get_encoding('gpt2')
    input_tokens = tokenizer.encode(prompt)
    x = torch.tensor(input_tokens).unsqueeze(0).repeat(num_return_sequences, 1)
    x = x.to(device)
    
    # Calculate final length (input length + requested additional tokens)
    input_length = x.size(1)
    target_length = input_length + max_length
    
    # Generate text
    with torch.no_grad():
        while x.size(1) < target_length:
            logits = model(x)[0]
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
    
    # Print token information once before generating sequences
    st.text(f"Size of Input tokens: {input_length}, Additional tokens to be predicted: {max_length}, Total tokens to be generated: {x.size(1)}")
    
    # Decode generated sequences
    generated_texts = []
    for i in range(num_return_sequences):
        tokens = x[i].tolist()
        text = tokenizer.decode(tokens)
        generated_texts.append(text)
    
    return generated_texts

# Streamlit UI
st.title("GPT Text Generator")

# Load model
model, device = load_model()

# Input form
prompt = st.text_area("Enter your prompt:", "Once upon a time")
max_length = st.slider("Predict additional text of length:", min_value=10, max_value=100, value=10)
num_sequences = st.slider("Number of sequences to generate:", 1, 5, 1)

if st.button("Generate"):
    with st.spinner("Generating text..."):
        generated_texts = generate_text(
            model=model,
            prompt=prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            device=device
        )
        
        # Display results
        for i, text in enumerate(generated_texts, 1):
            st.write(f"\nSequence {i}:")
            st.write(text) 