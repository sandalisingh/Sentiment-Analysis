# python3 ImportDSTweepy.py
 
# API FROM TWIITER to access tweets
import tweepy as tw

# here i have stored my api key and key secret 
from TweepyAuth_ESSENTIAL_ACCESS import API_KEY, API_KEY_SECRET

# AUTHENTICATION
auth = tw.OAuthHandler(API_KEY, API_KEY_SECRET)
API = tw.API(auth)

# 50 TWEETS 
tweets = API.search_tweets("covid", count=50);

# print(tweets);

