import os
import asyncio
import praw
from twscrape import API
from textblob import TextBlob
from typing import Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

class SocialSentiment:
    def __init__(self):
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Initialize Twitter API
        self.twitter_api = API()
        
        # Sentiment cache
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
    
    async def get_combined_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get combined sentiment from multiple sources"""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Fetch data concurrently
        reddit_sentiment = self.get_reddit_sentiment(symbol)
        twitter_sentiment = await self.get_twitter_sentiment(symbol)
        news_sentiment = await self.get_news_sentiment(symbol)
        
        # Combine results
        sentiment = {
            'reddit': reddit_sentiment,
            'twitter': twitter_sentiment,
            'news': news_sentiment,
            'combined': 0.4 * reddit_sentiment + 0.4 * twitter_sentiment + 0.2 * news_sentiment
        }
        
        # Update cache
        self.cache[cache_key] = sentiment
        return sentiment
    
    def get_reddit_sentiment(self, symbol: str) -> float:
        """Analyze Reddit sentiment"""
        try:
            submissions = self.reddit.subreddit('wallstreetbets').search(
                f'${symbol}', limit=50, time_filter='day')
            
            scores = []
            for submission in submissions:
                # Analyze title and comments
                title_score = TextBlob(submission.title).sentiment.polarity
                submission.comments.replace_more(limit=0)
                comment_scores = [TextBlob(c.body).sentiment.polarity 
                                for c in submission.comments.list()[:10]]
                scores.extend([title_score] + comment_scores)
            
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0
    
    async def get_twitter_sentiment(self, symbol: str) -> float:
        """Analyze Twitter sentiment"""
        try:
            tweets = await self.twitter_api.search(
                f"${symbol} lang:en", 
                limit=int(os.getenv('TWITTER_SCRAPE_LIMIT', 100)))
            
            scores = [TextBlob(t.rawContent).sentiment.polarity 
                     for t in tweets if not t.rawContent.startswith('RT')]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0
    
    async def get_news_sentiment(self, symbol: str) -> float:
        """Analyze news sentiment"""
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={os.getenv('NEWSAPI_KEY')}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    articles = data.get('articles', [])[:10]
                    scores = [TextBlob(a['title'] + " " + a.get('description', '')).sentiment.polarity 
                             for a in articles]
                    return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0