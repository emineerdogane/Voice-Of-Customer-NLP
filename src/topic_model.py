"""
BERTopic Topic Modeling Pipeline
Clusters app reviews into actionable product categories
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import clean_text


class ReviewTopicModeler:
    """Topic modeling for app reviews using BERTopic"""
    
    def __init__(self, language='english', min_topic_size=20):
        """
        Initialize topic modeler
        
        Args:
            language: Language for stop words
            min_topic_size: Minimum size of topics
        """
        self.language = language
        self.min_topic_size = min_topic_size
        self.model = None
        self.embeddings = None
        self.topic_labels = {}
        
    def prepare_data(self, df, text_column='review_text'):
        """
        Prepare and clean review data
        
        Args:
            df: DataFrame with reviews
            text_column: Column containing review text
            
        Returns:
            Cleaned DataFrame and list of documents
        """
        print("\n🧹 Cleaning review text...")
        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(clean_text)
        
        # Remove very short reviews
        df = df[df['cleaned_text'].str.len() >= 20].reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset='cleaned_text').reset_index(drop=True)
        
        print(f"✅ {len(df)} reviews ready for modeling")
        
        return df, df['cleaned_text'].tolist()
    
    def create_model(self):
        """Create BERTopic model with optimized settings"""
        
        # Use sentence transformers for embeddings
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Vectorizer for topic representations
        vectorizer_model = CountVectorizer(
            stop_words=self.language,
            min_df=2,
            ngram_range=(1, 2)
        )
        
        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=self.min_topic_size,
            nr_topics='auto',
            calculate_probabilities=True,
            verbose=True
        )
        
        return self.model
    
    def fit_model(self, documents):
        """
        Fit BERTopic model on documents
        
        Args:
            documents: List of review texts
            
        Returns:
            Topics and probabilities
        """
        print("\n🤖 Training BERTopic model...")
        print("This may take a few minutes...")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Fit model
        topics, probs = self.model.fit_transform(documents)
        
        print(f"\n✅ Model trained successfully!")
        print(f"📊 Found {len(set(topics)) - 1} topics (excluding outliers)")
        
        return topics, probs
    
    def label_topics_intelligently(self):
        """
        Automatically label topics based on keywords
        This converts technical topic numbers into actionable categories
        """
        print("\n🏷️ Labeling topics...")
        
        topic_info = self.model.get_topic_info()
        
        # Predefined category mappings based on common keywords
        category_keywords = {
            'Login/Authentication Bug': ['login', 'sign', 'password', 'account', 'authenticate', 'verify'],
            'Crash/Performance': ['crash', 'freeze', 'slow', 'lag', 'performance', 'loading', 'hang'],
            'UI/UX Issue': ['ui', 'interface', 'design', 'button', 'screen', 'layout', 'ugly'],
            'Feature Request': ['add', 'wish', 'want', 'need', 'missing', 'feature', 'would love'],
            'Billing/Payment': ['billing', 'payment', 'charge', 'subscription', 'refund', 'price', 'pay'],
            'Audio/Playback': ['audio', 'sound', 'play', 'music', 'volume', 'playback', 'song'],
            'Download/Offline': ['download', 'offline', 'storage', 'save', 'cache'],
            'Search/Discovery': ['search', 'find', 'discover', 'recommendation', 'browse'],
            'Ads/Premium': ['ad', 'premium', 'advertisement', 'skip', 'free'],
            'Syncing Issue': ['sync', 'device', 'library', 'playlist', 'update'],
        }
        
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                self.topic_labels[topic_id] = 'Outlier/Uncategorized'
                continue
            
            # Get top words for this topic
            top_words = [word for word, _ in self.model.get_topic(topic_id)[:10]]
            topic_text = ' '.join(top_words).lower()
            
            # Find best matching category
            best_match = 'Other'
            max_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for kw in keywords if kw in topic_text)
                if matches > max_matches:
                    max_matches = matches
                    best_match = category
            
            self.topic_labels[topic_id] = best_match
        
        # Print topic labels
        print("\n📋 Topic Categories:")
        for topic_id, label in sorted(self.topic_labels.items()):
            if topic_id != -1:
                count = len(topic_info[topic_info['Topic'] == topic_id])
                print(f"  {label}: {count} reviews")
    
    def assign_topics_to_df(self, df, topics, probs):
        """
        Add topic assignments to DataFrame
        
        Args:
            df: DataFrame with reviews
            topics: Topic assignments
            probs: Topic probabilities
            
        Returns:
            DataFrame with topic columns
        """
        df = df.copy()
        df['topic_id'] = topics
        df['topic_probability'] = [max(p) if isinstance(p, np.ndarray) else 0 for p in probs]
        df['topic_category'] = df['topic_id'].map(self.topic_labels)
        
        return df
    
    def save_model(self, output_dir='models'):
        """Save trained model and labels"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save BERTopic model
        model_path = os.path.join(output_dir, f'bertopic_model_{timestamp}')
        self.model.save(model_path)
        
        # Save topic labels
        labels_path = os.path.join(output_dir, f'topic_labels_{timestamp}.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(self.topic_labels, f)
        
        # Save as latest
        self.model.save(os.path.join(output_dir, 'bertopic_model_latest'))
        with open(os.path.join(output_dir, 'topic_labels_latest.pkl'), 'wb') as f:
            pickle.dump(self.topic_labels, f)
        
        print(f"\n💾 Model saved to: {output_dir}")
        
        return model_path
    
    def load_model(self, model_dir='models'):
        """Load saved model and labels"""
        model_path = os.path.join(model_dir, 'bertopic_model_latest')
        labels_path = os.path.join(model_dir, 'topic_labels_latest.pkl')
        
        self.model = BERTopic.load(model_path)
        with open(labels_path, 'rb') as f:
            self.topic_labels = pickle.load(f)
        
        print(f"✅ Model loaded from: {model_dir}")
    
    def visualize_topics(self, output_dir='visualizations'):
        """Generate topic visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n📊 Generating visualizations...")
        
        # 1. Topic distribution
        fig1 = self.model.visualize_barchart(top_n_topics=15)
        fig1.write_html(os.path.join(output_dir, 'topic_distribution.html'))
        
        # 2. Intertopic distance map
        fig2 = self.model.visualize_topics()
        fig2.write_html(os.path.join(output_dir, 'topic_map.html'))
        
        # 3. Topic hierarchy
        fig3 = self.model.visualize_hierarchy()
        fig3.write_html(os.path.join(output_dir, 'topic_hierarchy.html'))
        
        print(f"✅ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run BERTopic modeling on reviews')
    parser.add_argument('--input', type=str, default='data/raw/reviews_latest.csv',
                      help='Input CSV file with reviews')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                      help='Output directory for processed data')
    parser.add_argument('--min-topic-size', type=int, default=20,
                      help='Minimum topic size (default: 20)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\n📂 Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} reviews")
    
    # Initialize modeler
    modeler = ReviewTopicModeler(min_topic_size=args.min_topic_size)
    
    # Prepare data
    df_clean, documents = modeler.prepare_data(df)
    
    # Fit model
    topics, probs = modeler.fit_model(documents)
    
    # Label topics
    modeler.label_topics_intelligently()
    
    # Assign topics to dataframe
    df_final = modeler.assign_topics_to_df(df_clean, topics, probs)
    
    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'reviews_with_topics.csv')
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved processed data to: {output_path}")
    
    # Save model
    modeler.save_model()
    
    # Generate visualizations
    modeler.visualize_topics()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TOPIC MODELING SUMMARY")
    print("="*60)
    print(f"Total Reviews Processed: {len(df_final):,}")
    print(f"Number of Topics: {df_final['topic_category'].nunique()}")
    print("\nTop Categories:")
    print(df_final['topic_category'].value_counts().head(10).to_string())
    print("="*60)


if __name__ == '__main__':
    main()
