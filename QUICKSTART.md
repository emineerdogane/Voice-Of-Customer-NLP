# Quick Start Guide

## Project Setup

This guide will help you get the Voice of Customer analysis system up and running.

### 1. Install Dependencies

```bash
cd "voice_of_customer"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Scrape App Reviews

Choose an app to analyze (examples: Spotify, Uber, Instagram, etc.)

```bash
# Example: Scrape Spotify reviews
python src/scraper.py --app-id com.spotify.music --num-reviews 5000

# Example: Scrape Instagram reviews
python src/scraper.py --app-id com.instagram.android --num-reviews 5000

# Example: Scrape Uber reviews
python src/scraper.py --app-id com.ubercab --num-reviews 5000
```

**Output**: Reviews saved to `data/raw/reviews_latest.csv`

### 3. Run Topic Modeling

This step clusters reviews into actionable categories using BERTopic:

```bash
python src/topic_model.py
```

**What it does**:
- Cleans review text
- Trains BERTopic model on 5,000+ reviews
- Automatically categorizes reviews (Bugs, Feature Requests, UI Issues, etc.)
- Saves processed data to `data/processed/reviews_with_topics.csv`
- Generates interactive visualizations in `visualizations/` folder

**Note**: This step may take 5-10 minutes depending on your machine.

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Features

### 🔍 Filtering
- **Category Filter**: Focus on specific issue types (Bugs, UI, Billing, etc.)
- **Date Range**: Analyze trends over specific time periods
- **Rating Filter**: Filter by star ratings
- **Keyword Search**: Find reviews mentioning specific terms (e.g., "crash", "login")

### 📊 Analytics Tabs

1. **Distribution**: View category breakdown and rating distribution
2. **Trends**: Track how issues evolve over time (daily/weekly/monthly)
3. **Word Cloud**: Visualize most common terms
4. **Reviews**: Read actual user reviews with context

### 💡 Key Insights
- **Trending Issues**: Recent problems requiring attention
- **Popular Requests**: Most requested features

### 📥 Export Options
- Download filtered data as CSV
- Generate summary reports

## Use Cases for Product Managers

### 1. Track Bug Reports
```
Filter: Topic = "Crash/Performance"
Date Range: Last 30 days
Export: CSV for engineering team
```

### 2. Analyze Feature Requests
```
Keyword: "add" or "wish" or "want"
Filter: Rating >= 4
View: Trend over time
```

### 3. Monitor UI Feedback
```
Filter: Topic = "UI/UX Issue"
Compare: Before and after redesign
```

### 4. Billing Issues
```
Keyword: "charge" or "refund"
Filter: Rating <= 2
Action: Export for support team
```

## Project Structure

```
voice_of_customer/
├── data/
│   ├── raw/              # Scraped reviews (CSV)
│   └── processed/        # Reviews with topic labels
├── models/               # Saved BERTopic models
├── visualizations/       # Generated charts (HTML)
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── scraper.py        # Google Play Store scraper
│   ├── topic_model.py    # BERTopic clustering
│   └── utils.py          # Helper functions
└── app.py                # Streamlit dashboard
```

## Troubleshooting

### Issue: "Data file not found"
**Solution**: Make sure you've run the scraper and topic modeling steps first.

### Issue: Scraper fails
**Solution**: 
- Check internet connection
- Verify the app ID is correct (find it in Google Play Store URL)
- Try reducing the number of reviews

### Issue: Topic modeling is slow
**Solution**: 
- This is normal for first run (downloads ML models)
- Reduce `--min-topic-size` for faster processing
- Use a smaller sample of reviews for testing

### Issue: Dashboard won't load
**Solution**:
- Ensure Streamlit is installed: `pip install streamlit`
- Check that processed data exists: `data/processed/reviews_with_topics.csv`
- Try: `streamlit run app.py --server.port 8502`

## Advanced Usage

### Custom App Selection

Find app IDs from Google Play Store URLs:
```
https://play.google.com/store/apps/details?id=com.example.app
                                              ^^^^^^^^^^^^^^^^^^^^
                                              This is the app ID
```

### Adjust Topic Modeling

```bash
# More granular topics (smaller clusters)
python src/topic_model.py --min-topic-size 10

# Fewer, broader topics (larger clusters)
python src/topic_model.py --min-topic-size 50
```

### Analyze Multiple Apps

```bash
# Scrape different apps
python src/scraper.py --app-id com.spotify.music --num-reviews 3000
python src/scraper.py --app-id com.instagram.android --num-reviews 3000

# Combine data and run analysis
# Edit topic_model.py to load multiple files
```

## Resume Bullet Point

Use this accomplishment on your resume:

> "Engineered an NLP pipeline using BERTopic to cluster 5,000+ user reviews into actionable product categories (Bugs, UI, Billing); deployed a Streamlit web app allowing Product Managers to track feature-request sentiment in real-time."

## Next Steps

1. ✅ Scrape 5,000+ reviews
2. ✅ Run topic modeling
3. ✅ Explore dashboard
4. 📊 Create presentation with key insights
5. 📝 Document learnings in portfolio
6. 🚀 Add to GitHub with screenshots

## Tips for Showcasing

1. **Take screenshots** of the dashboard showing:
   - Topic distribution charts
   - Trend analysis
   - Keyword search results
   - Word clouds

2. **Prepare talking points**:
   - How you chose which app to analyze
   - Interesting insights discovered
   - How this helps product teams
   - Technical challenges overcome

3. **GitHub README should include**:
   - Project overview
   - Screenshots
   - Installation instructions
   - Sample outputs
   - Technologies used

Good luck with your project! 🚀
