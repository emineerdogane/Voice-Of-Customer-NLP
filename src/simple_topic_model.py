"""
Simple Topic Modeling without numba dependencies
Uses scikit-learn's LDA (Latent Dirichlet Allocation) instead of BERTopic
"""

import pandas as pd
import numpy as np
import re
import argparse
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle


def clean_text(text):
    """Clean review text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Extra spaces
    return text


def categorize_topic(topic_words):
    """Map topic words to actionable categories"""
    words = ' '.join(topic_words)
    
    # Bug/Error keywords
    if any(word in words for word in ['crash', 'bug', 'error', 'broken', 'fix', 'problem', 'issue', 'work', 'working']):
        return "🐛 Bugs & Errors"
    
    # Login/Account keywords
    if any(word in words for word in ['login', 'account', 'password', 'sign', 'authentication', 'log']):
        return "🔐 Login & Account Issues"
    
    # UI/UX keywords
    if any(word in words for word in ['interface', 'design', 'ui', 'ux', 'layout', 'menu', 'button', 'screen']):
        return "🎨 UI/UX Issues"
    
    # Feature Request keywords
    if any(word in words for word in ['feature', 'add', 'want', 'need', 'wish', 'request', 'would']):
        return "💡 Feature Requests"
    
    # Music/Audio keywords
    if any(word in words for word in ['music', 'song', 'audio', 'sound', 'quality', 'playlist', 'play', 'listen']):
        return "🎵 Music & Audio Quality"
    
    # Subscription/Billing keywords
    if any(word in words for word in ['subscription', 'premium', 'pay', 'billing', 'price', 'cost', 'free', 'ads']):
        return "💳 Subscription & Billing"
    
    # Performance keywords
    if any(word in words for word in ['slow', 'lag', 'performance', 'speed', 'loading', 'load', 'fast']):
        return "⚡ Performance Issues"
    
    # Positive feedback
    if any(word in words for word in ['love', 'great', 'amazing', 'best', 'perfect', 'excellent', 'good', 'awesome']):
        return "⭐ Positive Feedback"
    
    # App/Update related
    if any(word in words for word in ['app', 'update', 'version', 'new', 'change']):
        return "📱 App & Updates"
    
    return "📁 Other"


def main():
    parser = argparse.ArgumentParser(description='Simple Topic Modeling for Reviews')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--n_topics', type=int, default=10, help='Number of topics')
    parser.add_argument('--text_column', type=str, default='content', help='Column with review text')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 SIMPLE TOPIC MODELING")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df):,} reviews")
    
    # Check column names
    if args.text_column not in df.columns:
        # Try alternative column names
        for col in ['review_text', 'text', 'comment', 'content']:
            if col in df.columns:
                args.text_column = col
                break
    
    print(f"📝 Using column: '{args.text_column}'")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df['cleaned_text'] = df[args.text_column].apply(clean_text)
    
    # Filter
    df_clean = df[df['cleaned_text'].str.len() >= 20].copy()
    df_clean = df_clean.drop_duplicates(subset='cleaned_text').reset_index(drop=True)
    print(f"✅ {len(df_clean):,} clean reviews ready")
    
    # Vectorize
    print(f"\n🔤 Vectorizing text...")
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(df_clean['cleaned_text'])
    print(f"✅ Created {X.shape[1]} features")
    
    # LDA Topic Modeling
    print(f"\n🤖 Running LDA topic modeling with {args.n_topics} topics...")
    lda_model = LatentDirichletAllocation(
        n_components=args.n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online',
        verbose=1
    )
    
    doc_topics = lda_model.fit_transform(X)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Display topics
    print("\n" + "="*60)
    print("📊 DISCOVERED TOPICS")
    print("="*60)
    
    topic_labels = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        category = categorize_topic(top_words)
        topic_labels[topic_idx] = category
        
        print(f"\nTopic {topic_idx}: {category}")
        print(f"  Keywords: {', '.join(top_words[:7])}")
    
    # Assign topics to reviews
    print("\n📋 Assigning topics to reviews...")
    df_clean['topic'] = doc_topics.argmax(axis=1)
    df_clean['topic_category'] = df_clean['topic'].map(topic_labels)
    df_clean['topic_probability'] = doc_topics.max(axis=1)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'reviews_with_topics.csv')
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved results to: {output_path}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'lda_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'lda_model': lda_model,
            'vectorizer': vectorizer,
            'topic_labels': topic_labels
        }, f)
    print(f"💾 Saved model to: {model_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Total Reviews Processed: {len(df_clean):,}")
    print(f"Number of Categories: {df_clean['topic_category'].nunique()}")
    print("\n📈 Category Distribution:")
    print(df_clean['topic_category'].value_counts().to_string())
    
    # Sample reviews from each category
    print("\n" + "="*60)
    print("🔍 SAMPLE REVIEWS BY CATEGORY")
    print("="*60)
    
    for category in df_clean['topic_category'].value_counts().head(5).index:
        print(f"\n{category}")
        print("-" * 60)
        samples = df_clean[df_clean['topic_category'] == category].nlargest(2, 'topic_probability')
        for idx, row in samples.iterrows():
            score = row['score'] if 'score' in row else 'N/A'
            print(f"  ⭐ {score}/5: {row['cleaned_text'][:120]}...")
        print()
    
    print("="*60)
    print("✅ DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
