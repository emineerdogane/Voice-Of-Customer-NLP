"""
Google Play Store Review Scraper
Collects app reviews for Voice of Customer analysis
"""

import pandas as pd
import argparse
from google_play_scraper import app, Sort, reviews
from datetime import datetime
import time
from tqdm import tqdm
import os


class PlayStoreScraper:
    """Scraper for Google Play Store app reviews"""
    
    def __init__(self, app_id, lang='en', country='us'):
        """
        Initialize scraper
        
        Args:
            app_id: Google Play Store app ID (e.g., 'com.spotify.music')
            lang: Language code (default: 'en')
            country: Country code (default: 'us')
        """
        self.app_id = app_id
        self.lang = lang
        self.country = country
        self.app_info = None
        
    def get_app_info(self):
        """Fetch basic app information"""
        try:
            self.app_info = app(self.app_id, lang=self.lang, country=self.country)
            print(f"\n📱 App: {self.app_info['title']}")
            print(f"⭐ Rating: {self.app_info['score']}")
            print(f"📊 Reviews: {self.app_info['reviews']:,}")
            return self.app_info
        except Exception as e:
            print(f"❌ Error fetching app info: {e}")
            return None
    
    def scrape_reviews(self, num_reviews=5000, sort_by=Sort.MOST_RELEVANT):
        """
        Scrape reviews from Google Play Store
        
        Args:
            num_reviews: Target number of reviews to scrape
            sort_by: Sorting method (MOST_RELEVANT, NEWEST, RATING)
            
        Returns:
            DataFrame with reviews
        """
        print(f"\n🔍 Scraping {num_reviews} reviews from {self.app_id}...")
        
        all_reviews = []
        continuation_token = None
        
        # Calculate batches (200 reviews per batch)
        batch_size = 200
        num_batches = (num_reviews + batch_size - 1) // batch_size
        
        with tqdm(total=num_reviews, desc="Collecting reviews") as pbar:
            for batch in range(num_batches):
                try:
                    # Fetch reviews
                    result, continuation_token = reviews(
                        self.app_id,
                        lang=self.lang,
                        country=self.country,
                        sort=sort_by,
                        count=batch_size,
                        continuation_token=continuation_token
                    )
                    
                    all_reviews.extend(result)
                    pbar.update(len(result))
                    
                    # Stop if we've collected enough or no more reviews
                    if len(all_reviews) >= num_reviews or not continuation_token:
                        break
                    
                    # Be polite to the server
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"\n⚠️ Error in batch {batch}: {e}")
                    break
        
        print(f"\n✅ Collected {len(all_reviews)} reviews")
        return self._process_reviews(all_reviews[:num_reviews])
    
    def _process_reviews(self, reviews_data):
        """Convert reviews to DataFrame and clean data"""
        df = pd.DataFrame(reviews_data)
        
        # Select and rename key columns
        df = df[[
            'reviewId', 'userName', 'userImage', 'content', 'score',
            'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent'
        ]]
        
        df.rename(columns={
            'reviewId': 'review_id',
            'userName': 'user_name',
            'userImage': 'user_image',
            'content': 'review_text',
            'score': 'rating',
            'thumbsUpCount': 'thumbs_up',
            'reviewCreatedVersion': 'app_version',
            'at': 'review_date',
            'replyContent': 'developer_reply'
        }, inplace=True)
        
        # Convert date
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Add metadata
        df['app_id'] = self.app_id
        df['scraped_at'] = datetime.now()
        
        # Calculate text length
        df['text_length'] = df['review_text'].str.len()
        
        # Filter out very short reviews (less than 10 characters)
        df = df[df['text_length'] >= 10].reset_index(drop=True)
        
        return df
    
    def save_data(self, df, output_dir='data/raw'):
        """Save reviews to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        app_name = self.app_id.split('.')[-1]
        filename = f"{app_name}_reviews_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n💾 Saved to: {filepath}")
        
        # Also save as latest
        latest_path = os.path.join(output_dir, 'reviews_latest.csv')
        df.to_csv(latest_path, index=False, encoding='utf-8-sig')
        
        return filepath
    
    def print_summary(self, df):
        """Print data summary statistics"""
        print("\n" + "="*60)
        print("📊 DATA SUMMARY")
        print("="*60)
        print(f"Total Reviews: {len(df):,}")
        print(f"Date Range: {df['review_date'].min().date()} to {df['review_date'].max().date()}")
        print(f"\nRating Distribution:")
        print(df['rating'].value_counts().sort_index().to_string())
        print(f"\nAverage Rating: {df['rating'].mean():.2f}")
        print(f"Average Text Length: {df['text_length'].mean():.0f} characters")
        print(f"Reviews with Developer Reply: {df['developer_reply'].notna().sum():,} ({df['developer_reply'].notna().sum()/len(df)*100:.1f}%)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Scrape Google Play Store reviews')
    parser.add_argument('--app-id', type=str, default='com.spotify.music',
                      help='Google Play Store app ID (e.g., com.spotify.music)')
    parser.add_argument('--num-reviews', type=int, default=5000,
                      help='Number of reviews to scrape (default: 5000)')
    parser.add_argument('--lang', type=str, default='en',
                      help='Language code (default: en)')
    parser.add_argument('--country', type=str, default='us',
                      help='Country code (default: us)')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                      help='Output directory (default: data/raw)')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = PlayStoreScraper(args.app_id, lang=args.lang, country=args.country)
    
    # Get app info
    app_info = scraper.get_app_info()
    if not app_info:
        return
    
    # Scrape reviews
    df = scraper.scrape_reviews(num_reviews=args.num_reviews)
    
    # Save data
    scraper.save_data(df, output_dir=args.output_dir)
    
    # Print summary
    scraper.print_summary(df)


if __name__ == '__main__':
    main()
