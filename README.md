# Voice of Customer - Automated Review Categorization System

## Project Overview
An NLP pipeline that transforms unstructured app reviews into actionable product insights using BERTopic for topic modeling. This system clusters 5,000+ user reviews into meaningful categories (Bugs, Feature Requests, UI/UX issues) and provides a real-time dashboard for Product Managers.

## Key Features
- **Automated Data Collection**: Scrapes 5,000+ reviews from Google Play Store
- **Advanced Topic Modeling**: Uses BERTopic (not simple sentiment analysis) to identify themes
- **Actionable Categorization**: Clusters reviews into product-relevant buckets (Login Bugs, UI Clutter, Feature Requests, etc.)
- **Interactive Dashboard**: Streamlit web app for tracking trends and keyword searches
- **Trend Analysis**: Visualize how specific issues evolve over time

## Tech Stack
- **Data Collection**: `google-play-scraper`
- **NLP & Topic Modeling**: BERTopic, sentence-transformers
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib
- **Web App**: Streamlit

## Project Structure
```
voice_of_customer/
├── data/
│   ├── raw/              # Raw scraped reviews
│   └── processed/        # Cleaned and categorized data
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── scraper.py        # Google Play Store scraper
│   ├── topic_model.py    # BERTopic clustering pipeline
│   └── utils.py          # Helper functions
├── app.py                # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Scrape Reviews
```bash
python src/scraper.py --app-id com.spotify.music --num-reviews 5000
```

### 2. Run Topic Modeling
```bash
python src/topic_model.py --input data/raw/reviews.csv
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

## Dashboard Features
- **Keyword Search**: Find reviews mentioning specific terms (e.g., "crash", "login")
- **Topic Distribution**: See breakdown of review categories
- **Trend Analysis**: Track how topics evolve over time
- **Export Capabilities**: Download filtered results for engineering teams

## Resume Bullet Point
*"Engineered an NLP pipeline using BERTopic to cluster 5,000+ user reviews into actionable product categories (Bugs, UI, Billing); deployed a Streamlit web app allowing Product Managers to track feature-request sentiment in real-time."*

## Key Insights
This project demonstrates:
- Ability to work with messy, unstructured text data
- Converting user feedback into engineering-actionable tickets
- Building data products for non-technical stakeholders
- Real-world application of advanced NLP (beyond basic sentiment analysis)

## Future Enhancements
- Multi-language support
- Automated alert system for emerging issues
- Integration with JIRA/Linear for ticket creation
- Competitive analysis (compare multiple apps)

## Author
Emine - Data Science Portfolio Project

## License
MIT
