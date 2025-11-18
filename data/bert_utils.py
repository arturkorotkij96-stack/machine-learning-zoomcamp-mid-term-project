from typing import List
import pandas as pd
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

from data.env_config import MAX_LEN, MODEL_NAME


# 1. Setup Device (Optimized for MacBook Pro, falls back to CUDA/CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Apple Silicon (M1/M2/M3) acceleration
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device} for feature generation.")


# 2. Initialize Model and Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(device)
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model or tokenizer. Check your internet connection or package installation: {e}")
    exit()


def get_bert_features(texts: List[str]) -> np.ndarray:
    """
    Generates BERT features for a list of text strings in batches.
    We extract the embedding of the [CLS] token (the pooled output) 
    which is typically used as the feature vector for downstream tasks.
    """
    # Replace NaN or non-string values with an empty string
    texts = [str(t) if pd.notna(t) else '' for t in texts]
    
    # Tokenize the input texts
    encoded_inputs = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )

    # Move tensors to the selected device
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    with torch.no_grad():
        # Get the hidden states from the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # The pooled output is the vector of the [CLS] token, commonly used as the sequence representation.
    # Shape: (batch_size, hidden_size=768)
    pooled_output = outputs.pooler_output
    
    # Move the results back to CPU and convert to a numpy array
    return pooled_output.cpu().numpy()
