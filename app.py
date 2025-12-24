"""
Voice of Customer - Streamlit Dashboard
Interactive dashboard for Product Managers to explore app review topics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed review data with topics"""
    data_path = 'data/processed/reviews_with_topics.csv'
    
    if not os.path.exists(data_path):
        st.error(f"❌ Data file not found at {data_path}")
        st.info("💡 Please run the scraper and topic modeling first:\n"
                "1. `python src/scraper.py --app-id com.spotify.music --num-reviews 5000`\n"
                "2. `python src/topic_model.py`")
        st.stop()
    
    df = pd.read_csv(data_path)
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    return df

def create_topic_distribution_chart(df):
    """Create bar chart of topic distribution"""
    topic_counts = df['topic_category'].value_counts().reset_index()
    topic_counts.columns = ['Category', 'Count']
    
    fig = px.bar(
        topic_counts.head(10),
        x='Count',
        y='Category',
        orientation='h',
        title='📊 Top 10 Review Categories',
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_trend_chart(df, category=None, timeframe='M'):
    """Create time series chart of review trends"""
    df_filtered = df.copy()
    
    if category and category != 'All Categories':
        df_filtered = df_filtered[df_filtered['topic_category'] == category]
    
    # Group by time period
    df_filtered['period'] = df_filtered['review_date'].dt.to_period(timeframe).dt.to_timestamp()
    trend_data = df_filtered.groupby('period').size().reset_index(name='count')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_data['period'],
        y=trend_data['count'],
        mode='lines+markers',
        name='Reviews',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    title = f'📈 Review Trend Over Time'
    if category and category != 'All Categories':
        title += f' - {category}'
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Reviews',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_rating_distribution(df, category=None):
    """Create rating distribution chart"""
    df_filtered = df.copy()
    
    if category and category != 'All Categories':
        df_filtered = df_filtered[df_filtered['topic_category'] == category]
    
    rating_counts = df_filtered['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    
    fig = px.bar(
        rating_counts,
        x='Rating',
        y='Count',
        title='⭐ Rating Distribution',
        color='Rating',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(height=350)
    
    return fig

def create_wordcloud(df, category=None):
    """Generate word cloud for reviews"""
    df_filtered = df.copy()
    
    if category and category != 'All Categories':
        df_filtered = df_filtered[df_filtered['topic_category'] == category]
    
    text = ' '.join(df_filtered['cleaned_text'].dropna())
    
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Blues',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def display_sample_reviews(df, category=None, keyword=None, n=5):
    """Display sample reviews"""
    df_filtered = df.copy()
    
    if category and category != 'All Categories':
        df_filtered = df_filtered[df_filtered['topic_category'] == category]
    
    if keyword:
        df_filtered = df_filtered[
            df_filtered['cleaned_text'].str.lower().str.contains(keyword.lower(), na=False)
        ]
    
    if len(df_filtered) == 0:
        st.warning("No reviews found matching the criteria")
        return
    
    # Sort by thumbs up and recent date
    df_filtered = df_filtered.nlargest(n, 'thumbs_up')
    
    for idx, row in df_filtered.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{row['topic_category']}** | ⭐ {row['rating']}/5")
                st.write(row['review_text'][:500] + ('...' if len(row['review_text']) > 500 else ''))
            
            with col2:
                st.caption(f"📅 {row['review_date'].strftime('%Y-%m-%d')}")
                st.caption(f"👍 {row['thumbs_up']} helpful")
            
            st.divider()

def main():
    # Header
    st.markdown('<p class="main-header">📊 Voice of Customer Dashboard</p>', unsafe_allow_html=True)
    st.markdown("*Automated Review Categorization using BERTopic*")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Category filter
    categories = ['All Categories'] + sorted(df['topic_category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Date range filter
    min_date = df['review_date'].min().date()
    max_date = df['review_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Rating filter
    rating_filter = st.sidebar.multiselect(
        "Rating",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5]
    )
    
    # Keyword search
    keyword = st.sidebar.text_input("🔎 Keyword Search", placeholder="e.g., crash, login, slow")
    
    # Apply filters
    df_filtered = df.copy()
    
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['review_date'].dt.date >= date_range[0]) &
            (df_filtered['review_date'].dt.date <= date_range[1])
        ]
    
    df_filtered = df_filtered[df_filtered['rating'].isin(rating_filter)]
    
    if selected_category != 'All Categories':
        df_filtered = df_filtered[df_filtered['topic_category'] == selected_category]
    
    if keyword:
        df_filtered = df_filtered[
            df_filtered['cleaned_text'].str.lower().str.contains(keyword.lower(), na=False)
        ]
    
    # Key metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df_filtered):,}")
    
    with col2:
        avg_rating = df_filtered['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    
    with col3:
        unique_topics = df_filtered['topic_category'].nunique()
        st.metric("Categories", unique_topics)
    
    with col4:
        recent_reviews = len(df_filtered[df_filtered['review_date'] >= (df['review_date'].max() - timedelta(days=30))])
        st.metric("Last 30 Days", f"{recent_reviews:,}")
    
    # Visualizations
    st.header("📊 Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Trends", "☁️ Word Cloud", "📝 Reviews"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(create_topic_distribution_chart(df_filtered), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_rating_distribution(df_filtered, selected_category), use_container_width=True)
    
    with tab2:
        # Timeframe selector
        timeframe = st.radio(
            "Timeframe",
            options=['D', 'W', 'M'],
            format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
            horizontal=True
        )
        
        st.plotly_chart(
            create_trend_chart(df_filtered, selected_category, timeframe),
            use_container_width=True
        )
        
        # Category comparison
        if selected_category == 'All Categories':
            st.subheader("Category Trends")
            top_categories = df_filtered['topic_category'].value_counts().head(5).index
            
            trend_comparison = df_filtered[df_filtered['topic_category'].isin(top_categories)].copy()
            trend_comparison['period'] = trend_comparison['review_date'].dt.to_period(timeframe).dt.to_timestamp()
            
            trend_data = trend_comparison.groupby(['period', 'topic_category']).size().reset_index(name='count')
            
            fig = px.line(
                trend_data,
                x='period',
                y='count',
                color='topic_category',
                title='Top 5 Categories Over Time'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Word Cloud")
        wordcloud_fig = create_wordcloud(df_filtered, selected_category)
        
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
        else:
            st.info("Not enough data to generate word cloud")
    
    with tab4:
        st.subheader("Sample Reviews")
        
        num_reviews = st.slider("Number of reviews to display", 5, 20, 10)
        
        display_sample_reviews(df_filtered, selected_category, keyword, num_reviews)
    
    # Insights section
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔥 Trending Issues")
        
        # Get recent negative reviews
        recent_negative = df_filtered[
            (df_filtered['rating'] <= 2) &
            (df_filtered['review_date'] >= (df_filtered['review_date'].max() - timedelta(days=7)))
        ]
        
        if len(recent_negative) > 0:
            recent_issues = recent_negative['topic_category'].value_counts().head(3)
            for category, count in recent_issues.items():
                st.write(f"• **{category}**: {count} reviews in last 7 days")
        else:
            st.write("No significant issues in the last 7 days")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("✨ Popular Requests")
        
        # Feature requests with high ratings
        feature_requests = df_filtered[
            (df_filtered['topic_category'].str.contains('Feature', case=False, na=False))
        ]
        
        if len(feature_requests) > 0:
            st.write(f"• **{len(feature_requests):,}** feature request reviews")
            st.write(f"• Average sentiment: {feature_requests['rating'].mean():.1f}⭐")
            st.write(f"• Most recent: {feature_requests['review_date'].max().strftime('%Y-%m-%d')}")
        else:
            st.write("No feature requests in filtered data")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Export functionality
    st.header("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"voc_reviews_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary report
        summary = f"""
Voice of Customer Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Total Reviews: {len(df_filtered):,}
Average Rating: {df_filtered['rating'].mean():.2f}
Date Range: {df_filtered['review_date'].min().strftime('%Y-%m-%d')} to {df_filtered['review_date'].max().strftime('%Y-%m-%d')}

Top Categories:
{df_filtered['topic_category'].value_counts().head(5).to_string()}

Rating Distribution:
{df_filtered['rating'].value_counts().sort_index().to_string()}
        """
        
        st.download_button(
            label="📊 Download Summary Report (TXT)",
            data=summary,
            file_name=f"voc_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # Footer
    st.divider()
    st.caption("Built with ❤️ using BERTopic and Streamlit | Voice of Customer Analytics System")

if __name__ == '__main__':
    main()
