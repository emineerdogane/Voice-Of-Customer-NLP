"""
Example script showing how to use the Voice of Customer pipeline
"""

from src.scraper import PlayStoreScraper
from src.topic_model import ReviewTopicModeler
import pandas as pd

# Step 1: Scrape reviews
print("="*60)
print("STEP 1: Scraping Reviews")
print("="*60)

scraper = PlayStoreScraper(app_id='com.spotify.music', lang='en', country='us')
scraper.get_app_info()
df_reviews = scraper.scrape_reviews(num_reviews=1000)  # Small sample for demo
scraper.save_data(df_reviews, output_dir='data/raw')
scraper.print_summary(df_reviews)

# Step 2: Topic Modeling
print("\n" + "="*60)
print("STEP 2: Topic Modeling")
print("="*60)

modeler = ReviewTopicModeler(min_topic_size=15)
df_clean, documents = modeler.prepare_data(df_reviews)
topics, probs = modeler.fit_model(documents)
modeler.label_topics_intelligently()

# Step 3: Assign topics and save
print("\n" + "="*60)
print("STEP 3: Processing Results")
print("="*60)

df_final = modeler.assign_topics_to_df(df_clean, topics, probs)
df_final.to_csv('data/processed/reviews_with_topics.csv', index=False)
modeler.save_model()

# Step 4: Quick analysis
print("\n" + "="*60)
print("STEP 4: Quick Analysis")
print("="*60)

print("\nTop 5 Categories:")
print(df_final['topic_category'].value_counts().head())

print("\nAverage rating by category:")
print(df_final.groupby('topic_category')['rating'].mean().sort_values())

print("\n✅ Pipeline complete! Run 'streamlit run app.py' to view dashboard")
