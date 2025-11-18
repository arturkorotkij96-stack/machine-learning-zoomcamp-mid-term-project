import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

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
    # Initialize English stop words set
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))
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
