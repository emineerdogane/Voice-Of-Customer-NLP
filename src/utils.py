"""
Utility functions for text preprocessing and analysis
"""

import re
import pandas as pd
from typing import List
import numpy as np


def clean_text(text):
    """
    Clean and preprocess review text
    
    Args:
        text: Raw review text
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def categorize_rating(rating):
    """
    Categorize rating into sentiment
    
    Args:
        rating: Star rating (1-5)
        
    Returns:
        Sentiment category
    """
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'


def extract_keywords(text, keywords):
    """
    Check if text contains any of the specified keywords
    
    Args:
        text: Review text
        keywords: List of keywords to search for
        
    Returns:
        List of found keywords
    """
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    return found


def calculate_topic_trends(df, date_column='review_date', topic_column='topic', freq='M'):
    """
    Calculate topic trends over time
    
    Args:
        df: DataFrame with reviews
        date_column: Name of date column
        topic_column: Name of topic column
        freq: Frequency for grouping ('D', 'W', 'M')
        
    Returns:
        DataFrame with topic counts over time
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create period column
    df['period'] = df[date_column].dt.to_period(freq)
    
    # Count topics per period
    trends = df.groupby(['period', topic_column]).size().reset_index(name='count')
    trends['period'] = trends['period'].dt.to_timestamp()
    
    return trends


def get_topic_summary(df, topic_column='topic', min_samples=10):
    """
    Generate summary statistics for each topic
    
    Args:
        df: DataFrame with reviews and topics
        topic_column: Name of topic column
        min_samples: Minimum samples required to include topic
        
    Returns:
        DataFrame with topic summaries
    """
    summaries = []
    
    for topic in df[topic_column].unique():
        topic_df = df[df[topic_column] == topic]
        
        if len(topic_df) < min_samples:
            continue
        
        summary = {
            'topic': topic,
            'count': len(topic_df),
            'avg_rating': topic_df['rating'].mean(),
            'recent_count': len(topic_df[topic_df['review_date'] >= topic_df['review_date'].max() - pd.Timedelta(days=30)]),
            'sample_reviews': topic_df.nlargest(3, 'thumbs_up')['review_text'].tolist()
        }
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries).sort_values('count', ascending=False)


def filter_by_keyword(df, keyword, text_column='review_text'):
    """
    Filter reviews by keyword
    
    Args:
        df: DataFrame with reviews
        keyword: Keyword to search for
        text_column: Column to search in
        
    Returns:
        Filtered DataFrame
    """
    mask = df[text_column].str.lower().str.contains(keyword.lower(), na=False)
    return df[mask]


def get_representative_docs(df, topic_column='topic', text_column='review_text', n=5):
    """
    Get most representative documents for each topic
    
    Args:
        df: DataFrame with reviews and topics
        topic_column: Name of topic column
        text_column: Name of text column
        n: Number of representative docs per topic
        
    Returns:
        Dictionary mapping topics to representative reviews
    """
    representatives = {}
    
    for topic in df[topic_column].unique():
        topic_df = df[df[topic_column] == topic]
        
        # Get reviews with most thumbs up
        top_reviews = topic_df.nlargest(n, 'thumbs_up')[text_column].tolist()
        representatives[topic] = top_reviews
    
    return representatives
