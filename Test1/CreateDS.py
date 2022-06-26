# python3 Test1/CreateDS.py

from ImportDSTweepy import tweets
import re
import pandas as pd 

DATAFRAME = pd.DataFrame(columns=(
    'id', 'tweet_text', 'truncated', 'followers_count', 'friends_count', 'user_account_age',
    'user_verified', 'favourites_count', 'user_tweets', 'tweet_retweeted',  'created_at', 
    # 'country_code', 
    'retweet_count', 'favorite_count', 'lang'
));

def extract_country_code(row): 
    if row.place:         
        return row.place.country_code    
    else:         
        return None

# REMOVE DUPLICATES

DATAFRAME.sort_values("tweet_text", inplace=True)
DATAFRAME.drop_duplicates(subset="tweet_text", keep=False, inplace=True)

for tweet in tweets:
    DATAFRAME = DATAFRAME.append({
        'id': tweet.id,
        'tweet_text' : re.sub(r'http\S+','', tweet.text),    
        # removing urls (re.sub->subsitutes url with space in tweets text)
        # starts with http until a space is reqd (ie S+)
        'truncated': tweet.truncated,
        'followers_count': tweet.user.followers_count,
        'friends_count' : tweet.user.friends_count,
        'user_account_age': tweet.user.created_at,
        'user_verified': tweet.user.verified,
        'favourites_count':tweet.user.favourites_count,
        'user_tweets': tweet.user.statuses_count,
        'tweet_retweeted': tweet.retweeted,
        'lang' : tweet.lang,
        'created_at': tweet.created_at,
        # 'country_code': extract_country_code(tweet),
        # 'country_code' : tweet.author.derived.locations.country,
        # country code not available in essential access
        'retweet_count': tweet.retweet_count,
        'favorite_count': tweet.favorite_count
    }, ignore_index=True)

print(DATAFRAME.head(50));

