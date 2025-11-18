from pathlib import Path
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from typing import List
from tqdm import tqdm
import numpy as np
import os
# Import for dimensionality reduction
from sklearn.decomposition import PCA 
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textstat.textstat import textstat
import time
import ssl # <-- ADDED IMPORT FOR SSL BYPASS

# --- ADDED IMPORTS FOR TOPIC MODELING (LDA) ---
from sklearn.decomposition import LatentDirichletAllocation

from data.bert_utils import get_bert_features
from data.env_config import BATCH_SIZE, DATASET_ROW_COUNT, INPUT_FILE, MAX_DF, MIN_DF, N_FEATURES, NUM_TOPICS, OUTPUT_FILE, TEXT_COLUMN
from data.lda_utils import extract_lda_topics
from data.ntlk_utils import extract_nlp_features, ntlk_init
from midterm_project.conf import Y_COL_RAW


def get_nlp_features_v1(df):
    """
    Run sentiment anlysys and extract some basic features with extract_nlp_features
    Then discover text topics with LDA analisys using CountVectorizer, and LatentDirichletAllocation
    """
    # VADER SentimentIntensityAnalyzer and flesch_kincaid_grade
    nlp_features_df = df.apply(
        extract_nlp_features, 
        axis=1, 
        result_type='expand'
    )

    # LDA topics discovering
    lda_features_df = extract_lda_topics(
        df, 
        TEXT_COLUMN, 
        NUM_TOPICS, 
        MAX_DF, 
        MIN_DF
    )

    return nlp_features_df, lda_features_df


def main():
    """
    Main function to load data, process, and save features.
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

    # Load the dataset and rows with empty score value
    try:
        df = pd.read_csv(INPUT_FILE)
        df[Y_COL_RAW] = pd.to_numeric(df[Y_COL_RAW], errors="coerce")
        df = df[df[Y_COL_RAW].notna()]
        df = df.sample(n=DATASET_ROW_COUNT, random_state=42)

        print(f"Successfully loaded dataset with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if the text column exists
    if TEXT_COLUMN not in df.columns:
        raise ValueError("Column '{TEXT_COLUMN}' not found. Please check configuration and run script again")

    text_data = df[TEXT_COLUMN].tolist()

    print("Building basic features....")
    nlp_features_df, lda_features_df = get_nlp_features_v1(df)

    all_features = []
    
    # Build bert features
    num_samples = len(text_data)
    for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Generating Features (768 Dim)"):
        batch = text_data[i:i + BATCH_SIZE]
        features = get_bert_features(batch)
        all_features.append(features)

    # Concatenate all batches into a single numpy array (Shape: N x 768)
    final_features = np.concatenate(all_features, axis=0)
    
    # Reduce dimencion to N_FEATURES with PCA ---
    print(f"\nApplying Principal Component Analysis (PCA) to reduce features from 768 to {N_FEATURES}...")
    
    # Handle the case where the dataset might be smaller than N_FEATURES
    n_components_actual = min(N_FEATURES, final_features.shape[0], final_features.shape[1])
    if n_components_actual < N_FEATURES:
        print(f"Warning: Cannot reduce to {N_FEATURES} components. Using {n_components_actual} instead.")

    # Initialize PCA
    pca = PCA(n_components=n_components_actual)
    
    # Fit PCA on the full 768-dimensional features and transform them
    reduced_features = pca.fit_transform(final_features)
    
    # Convert numpy array to DataFrame. Each of the N_FEATURES dimensions gets its own column.
    feature_df = pd.DataFrame(reduced_features, columns=[f"berd_feature_{i}" for i in range(n_components_actual)])
    res_df = pd.concat([df, feature_df, nlp_features_df, lda_features_df], axis=1)

    # Save the DataFrame to CSV
    res_df.to_csv(OUTPUT_FILE, index=False)    
    print(f"Feature generation and reduction complete!")
    print(f"Total rows processed: {feature_df.shape[0]}")
    print(f"Final feature shape: {feature_df.shape} ({feature_df.shape[1]} dimensions per sample)")
    print(f"Features saved to: {OUTPUT_FILE} (CSV format)")


if __name__ == "__main__":
    ntlk_init()

    main()
