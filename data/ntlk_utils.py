import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from textstat.textstat import textstat

from data.env_config import TEXT_COLUMN

def ntlk_init():
    """
    Download required packages for ntlk
    """
    # Attempt to download resources quietly
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab')

try:


    # Initialize VADER sentiment analyzer
    # Imports are placed here to catch potential LookupErrors during initialization 
    # if the required resource files are missing.
    VADER = SentimentIntensityAnalyzer()

except Exception as e:
    print(f"NLTK download failed (this is often due to network restrictions). Please run 'nltk.download(\"punkt\")' and 'nltk.download(\"vader_lexicon\")' manually. Error: {e}")


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

    # NLTK tokenization (requires 'punkt')
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)

        
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