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
from textstat.textstat import textstat
import time
import ssl # <-- ADDED IMPORT FOR SSL BYPASS

# --- ADDED IMPORTS FOR TOPIC MODELING (LDA) ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from midterm_project.conf import Y_COL_RAW


try:
    # --- START SSL CERTIFICATE VERIFICATION BYPASS (Fixes 'CERTIFICATE_VERIFY_FAILED') ---
    # This temporarily sets the default SSL context to an unverified one, allowing NLTK 
    # to download resources even if the system's root certificates are not properly configured.
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python does not have the context
        pass
    else:
        # Sideload the unverified context
        ssl._create_default_https_context = _create_unverified_https_context

    # --- END SSL BYPASS ---
    # Attempt to download resources quietly
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab')

    # Initialize VADER sentiment analyzer
    # Imports are placed here to catch potential LookupErrors during initialization 
    # if the required resource files are missing.
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
    
    # Initialize English stop words set
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception as e:
    print(f"NLTK download failed (this is often due to network restrictions). Please run 'nltk.download(\"punkt\")' and 'nltk.download(\"vader_lexicon\")' manually. Error: {e}")


# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 300
PROJECT_PATH  = Path(__file__).resolve().parent
INPUT_FILE = f'{PROJECT_PATH}/raw/Train_v2.csv' 
OUTPUT_FILE = 'processed_data_256_dim.csv' # Updated output file name
TEXT_COLUMN = 'body' # Assuming the main text content is in a column named 'body'

# --- GLOBAL CONFIGURATION CONSTANTS (Moved outside main) ---
NUM_TOPICS = 30 # Number of topics to discover (can be tuned)
MAX_DF = 0.95 # Ignore terms that appear in more than 95% of the documents
MIN_DF = 2

# New configuration for dimensionality reduction
N_FEATURES = 256 # Set this to your desired number of features!

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

def extract_nlp_features(row):
    """
    Extracts various NLP features from a single text string.
    
    Args:
        row (pd.Series): The row of the DataFrame being processed (required for axis=1).
        
    Returns:
        dict: A dictionary of extracted features.
    """
    # Extract the text from the row using the global TEXT_COLUMN constant
    text = row[TEXT_COLUMN] 

    # Ensure text is a string, handling NaN or other non-string data
    if pd.isna(text) or not isinstance(text, str):
        text = ""

    # --- Robust Tokenization and Counts ---
    try:
        # Preferred NLTK tokenization (requires 'punkt')
        words = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback if NLTK resources (like 'punkt') are missing or fail to load
        # Basic word tokenization (find all sequences of letters/numbers)
        words = [w for w in re.findall(r'\b\w+\b', text.lower())]
        # Basic sentence tokenization (split by common end punctuation)
        sentences = re.split(r'[.?!]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)

    # Calculate features, handling division by zero for empty strings
    
    # 1. Length and Average Features
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # 2. Lexical Complexity
    # Type-Token Ratio (TTR): measures vocabulary richness
    unique_words = set(w.lower() for w in words if w.isalnum())
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    
    # 3. Sentiment (VADER) - Uses conditional check
    if VADER:
        sentiment_scores = VADER.polarity_scores(text)
    else:
        # Default to neutral if VADER is not initialized
        sentiment_scores = {'compound': 0.0, 'neg': 0.0, 'pos': 0.0}

    # 4. Readability (Flesch-Kincaid Grade Level)
    # The grade level indicates the reading difficulty of the text.
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
    except:
        # Assign 0 if textstat fails (e.g., text is too short or empty)
        fk_grade = 0
        
    # 5. Character/Punctuation Features
    # Count of characters that are entirely capitalized (potential shouting)
    all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    # Count of special characters (e.g., for emphasis or questions)
    exclamation_count = text.count('!')
    question_mark_count = text.count('?')
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'ttr': ttr,
        'fk_grade': fk_grade,
        'vader_compound': sentiment_scores['compound'],
        'vader_neg': sentiment_scores['neg'],
        'vader_pos': sentiment_scores['pos'],
        'all_caps_word_count': all_caps_words,
        'exclamation_count': exclamation_count,
        'question_mark_count': question_mark_count
    }


def display_topics(model, feature_names, no_top_words):
    """
    Prints the top words for each discovered topic.
    """
    topic_summary = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([feature_names[i]
                              for i in topic.argsort()[:-no_top_words - 1:-1]])
        topic_summary.append(f"Topic {topic_idx + 1}: {top_words}")
    return "\n".join(topic_summary)

def extract_lda_topics(df, text_column, num_topics, max_df, min_df):
    """
    Performs Latent Dirichlet Allocation (LDA) for topic modeling across the corpus.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text.
        num_topics (int): The number of topics to extract.
        max_df (float): Max document frequency for CountVectorizer.
        min_df (int): Min document frequency for CountVectorizer.
        
    Returns:
        pd.DataFrame: A DataFrame with columns for topic probabilities (e.g., 'topic_0', 'topic_1', ...).
    """
    
    # 1. Preprocess and Vectorize Text
    print("\nStarting LDA Topic Modeling...")
    
    # Fill NaN values with an empty string to prevent CountVectorizer error
    corpus = df[text_column].fillna('').astype(str).tolist()
    
    # If STOP_WORDS is an empty set (due to failed NLTK init), we pass None 
    # to CountVectorizer's stop_words parameter to maintain correct behavior.
    stop_words_param = list(STOP_WORDS) if STOP_WORDS else None
    vectorizer = CountVectorizer(
        max_df=max_df, 
        min_df=min_df, 
        stop_words=stop_words_param,
        token_pattern=r'\b[a-zA-Z]{3,}\b' # Only words with 3 or more letters
    )
    
    # Fit the vectorizer to the data and transform the data into a DTM
    dtm = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    # 2. Fit the LDA Model
    lda = LatentDirichletAllocation(
        n_components=num_topics, 
        random_state=42, 
        learning_method='online', # Faster and better for large datasets
        max_iter=5
    )
    
    # Fit the model and transform the DTM to get document-topic probabilities
    doc_topic_matrix = lda.fit_transform(dtm)
    
    # 3. Output Topic Summary
    print(f"LDA Model fit complete with {num_topics} topics.")
    print("\n--- TOPIC INTERPRETATION (Top 10 words per topic) ---")
    print(display_topics(lda, feature_names, 10))
    print("-----------------------------------------------------")
    
    # 4. Create Feature DataFrame
    topic_columns = [f'lda_topic_{i}' for i in range(num_topics)]
    topic_df = pd.DataFrame(doc_topic_matrix, columns=topic_columns, index=df.index)
    
    return topic_df

def get_nlp_features_v1(df):
    print(f"Data loaded successfully. Total rows: {len(df)}")
    print(f"Extracting features from column: '{TEXT_COLUMN}'...")
    
    # Use df.apply(..., axis=1, result_type='expand') to call the function row-wise,
    # expanding the returned dictionary into new columns and returning a DataFrame.
    output_columns = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'ttr',
       'fk_grade', 'vader_compound', 'vader_neg', 'vader_pos',
       'all_caps_word_count', 'exclamation_count', 'question_mark_count']
    nlp_features_df = df.apply(
        extract_nlp_features, 
        axis=1, 
        result_type='expand'
    )
    df.sort_values
    # Join the new features back to the original DataFrame
    # df = pd.concat([df, new_features], axis=1)
    # Note: LDA runs on the *whole* corpus, not row-by-row.
    lda_features_df = extract_lda_topics(
        df, 
        TEXT_COLUMN, 
        NUM_TOPICS, 
        MAX_DF, 
        MIN_DF
    )

    return nlp_features_df, lda_features_df

def main():
    """Main function to load data, process, and save features."""
    if not os.path.exists(INPUT_FILE):
        print(f"🚨 Error: Input file '{INPUT_FILE}' not found.")
        print(f"Please download the 'Train_v2.csv' file from Kaggle and place it in the same directory as this script.")
        return

    # Load the dataset
    try:
        df = pd.read_csv(INPUT_FILE)
        df[Y_COL_RAW] = pd.to_numeric(df[Y_COL_RAW], errors="coerce")
        df = df[df[Y_COL_RAW].notna()]
        print(f"Successfully loaded dataset with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if the assumed text column exists, and choose an alternative if necessary
    if TEXT_COLUMN not in df.columns:
        print(f"Warning: Column '{TEXT_COLUMN}' not found. Looking for 'title'...")
        if 'title' in df.columns:
            text_data = df['title'].tolist()
            print(f"Using column 'title' for feature generation.")
        else:
            print(f"Error: Neither '{TEXT_COLUMN}' nor 'title' found. Please update TEXT_COLUMN to your actual text column name.")
            return
    else:
        text_data = df[TEXT_COLUMN].tolist()
        print(f"Using column '{TEXT_COLUMN}' for feature generation.")

    print("Building basic features....")
    nlp_features_df, lda_features_df = get_nlp_features_v1(df)

    all_features = []
    
    # Process data in batches
    num_samples = len(text_data)
    for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Generating Features (768 Dim)"):
        batch = text_data[i:i + BATCH_SIZE]
        features = get_bert_features(batch)
        all_features.append(features)

    # Concatenate all batches into a single numpy array (Shape: N x 768)
    final_features = np.concatenate(all_features, axis=0)
    
    # --- New logic for Dimensionality Reduction (PCA) ---
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
    # --- End new saving logic ---
    
    print("\n" + "="*50)
    print(f"✅ Feature generation and reduction complete!")
    print(f"Total rows processed: {feature_df.shape[0]}")
    print(f"Final feature shape: {feature_df.shape} ({feature_df.shape[1]} dimensions per sample)")
    print(f"Features saved to: {OUTPUT_FILE} (CSV format)")
    print("="*50)


if __name__ == "__main__":
    main()