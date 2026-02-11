import tweepy
import pandas as pd
import re
from emoji import demojize

# -------------------------
# ADD YOUR TWITTER API KEYS HERE
# -------------------------
API_KEY = "YOUR_API_KEY"
API_SECRET_KEY = "YOUR_API_SECRET_KEY"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
ACCESS_TOKEN_SECRET = "YOUR_ACCESS_SECRET"
BEARER_TOKEN = "YOUR_BEARER_TOKEN"

# -------------------------
# Authenticate Tweepy (v2)
# -------------------------
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# -------------------------
# Preprocess Function
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = demojize(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.strip()
    return text

# -------------------------
# Fetch Tweets Function
# -------------------------
def fetch_tweets(query, max_tweets=200):
    tweets_list = []
    for tweet in tweepy.Paginator(client.search_recent_tweets,
                                  query=query,
                                  tweet_fields=['text','created_at','lang'],
                                  max_results=100).flatten(limit=max_tweets):
        if tweet.lang in ['en','und']:  # Tanglish is often undetermined language
            tweets_list.append({
                "text": tweet.text,
                "clean_text": preprocess_text(tweet.text),
                "created_at": tweet.created_at
            })
    df = pd.DataFrame(tweets_list)
    return df

# -------------------------
# Example Queries
# -------------------------
queries = ["enna da", "super da", "chumma da", "tension da", "enakku", "naane"]
all_tweets = pd.DataFrame()

for q in queries:
    df = fetch_tweets(q, max_tweets=200)
    all_tweets = pd.concat([all_tweets, df], ignore_index=True)

# Save to CSV
all_tweets.to_csv("../data/tanglish_tweets.csv", index=False)
print("Fetched tweets saved successfully!")
