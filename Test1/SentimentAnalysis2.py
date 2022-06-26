# python3 Test1/SentimentAnalysis2.py

#--------------------------------------- I M P O R T I N G   P A C K A G E S
import numpy as np #to perform a wide variety of mathematical operations on arrays.
import pandas as pd #used for data analysis and associated manipulation of tabular data in Dataframes. 
import matplotlib.pyplot as plt #plotting library
import seaborn as sns   #a data visualization library built on top of matplotlib and closely integrated with pandas data structures in Python.
from textblob import TextBlob  
from textblob import Word 
import nltk
import re   #Regular Expression, is a sequence of characters that forms a search pattern.
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from nltk.stem import WordNetLemmatizer

# from StopWordsList import stopwords

#--------------------------------------- R E A D I N G   D A T A
from CreateDS import DATAFRAME

print("\n\n---------------------->> HEAD (initially) <<----------------------\n\n")
print(DATAFRAME.head())

#--------------------------------------- D I S P L A Y   C O L U M N S
print("\n\n---------------------->> COLS (initially) <<----------------------\n\n")
print(DATAFRAME.columns)

#---------------------------------------C L E A N N I N G  D A T A 
# : DROPPING  COLS
TweetsDS = DATAFRAME.copy()
TweetsDS.drop([
        'truncated', 'followers_count', 'friends_count', 'user_account_age',
        'user_verified', 'user_tweets', 'favorite_count',
        'tweet_retweeted'
    ], axis=1, inplace = True
    # axis = 1 means cols & axis = 0 means rows
    # inplace = true means data updated in dataframe & doesn't return anything
)
print("\n\n---------------------->> COLS (after droppin irreevant cols) <<----------------------\n\n")
print(TweetsDS.columns)
print("\n\n---------------------->> HEAD (after droppin irreevant cols) <<----------------------\n\n")
print(TweetsDS.head())

# OnlyCountryAndLangCols = TweetsDS[['country_code', 'lang']]
print("\n\n---------------------->> LANG COL (initially) <<----------------------\n\n")
print( TweetsDS[['lang']].head(10) )

# : FILTERING DATA WITH 'country_code = IN' & 'language = en'
TweetsDS = TweetsDS[( TweetsDS.lang == "en" )].reset_index(drop=True);
print("\n\n---------------------->> LANG COL ( keeping en LANG ) <<----------------------\n\n")
print(TweetsDS[['lang']].head(10))

TweetsDS.drop([
        'lang'
    ], axis=1, inplace = True
    # axis = 1 means cols & axis = 0 means rows
    # inplace = true means data updated in dataframe & doesn't return anything
)

print("\n\n---------------------->> SHAPE (after dropping language Cols) <<----------------------\n\n")
print(TweetsDS.shape)

print("\n\n---------------------->> HEAD (after dropping language(en) Cols) <<----------------------\n\n")
print(TweetsDS.head(10))

# TweetsDS.reset_index();
# print("\n\n---------------------->> HEAD (after resetting indices) <<----------------------\n\n")
# print(TweetsDS.head(10))

# SHAPE
print("\n\n---------------------->> SHAPE (finally) <<----------------------\n\n")
print(TweetsDS.shape)

# CHECK MISSING VALUES
print("\n\n---------------------->> NO OF MISSING VALUES IN EACH COL <<----------------------\n\n")
print(TweetsDS.isna().sum())

# DATA PREPROCESSING
print("\n\n---------------------->> TEXT (before preprocessing) <<----------------------\n\n")
print(TweetsDS['tweet_text'].head())

    # REF FOR CREATING RE EQUATION
    # ()	            grps
    # [a-zA-Z]	        any digit 0-9 or character a-z or A-Z
    # [^0-9A-Za-z \t]	(2) any character except digit(0-9)/char(a-z or A-Z)/tab
    # |	                Either or
    # \S                not a whitespace character.
    # \S+               extracts chars till a space encountered
    # \w                word
    # \w+               extract words till space encountered
    # (\w+:\/\/\S+) -> :// -> https:// --> (3) removing links
    # (@[A-Za-z0-9]+)   @ (chars/digits) till space encounteres
    #                   -> remove @username (1)
    # (#[A-Za-z0-9]+) -> # (digit/char) till space encountered
    #                 -> (4) removea #hashtags

for i in range(TweetsDS.shape[0]) :
    TweetsDS['tweet_text'][i] = ' '.join(
        re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", 
            " ", TweetsDS['tweet_text'][i]
        ).split()
    ).lower()
print("\n\n---------------------->> TEXT (after preprocessing) <<----------------------\n\n")
print("REMOVING -  https://links, #hashtags, @username, anything other than english words and numbers \n\n")
print(TweetsDS['tweet_text'].head())


#--------------------------------------- T O P  5  M O S T  F A V  T W E E T S
fav = TweetsDS[['favourites_count','tweet_text']].sort_values('favourites_count',ascending = False)[:5].reset_index()
print("\n\n---------------------->> TOP 5 MOST FAV TWEETS <<----------------------\n\n")
for i in range(5):
    print(i,']', fav['tweet_text'][i],'\n')

#--------------------------------------- T O P  5  M O S T  R E T W E E T E D  T W E E T S
retweet = TweetsDS[['retweet_count','tweet_text']].sort_values('retweet_count',ascending = False)[:5].reset_index()
print("\n\n---------------------->> TOP 5 MOST RETWEETED TWEETS <<----------------------\n\n")
for i in range(5):
    print(i,']', retweet['tweet_text'][i],'\n')

#--------------------------------------- R E M O V I N G  S T O P  W O R D S
from StopWordsList import stopwords
print("\n\n---------------------->> HEAD ( before removing STOPWORDS) <<----------------------\n\n")
print(TweetsDS['tweet_text'].head())
TweetsDS['tweet_text'] = TweetsDS['tweet_text'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))
print("\n\n---------------------->> HEAD ( after removing STOPWORDS) <<----------------------\n\n")
print(TweetsDS['tweet_text'].head())

for i in range(TweetsDS.shape[0]) :
    TweetsDS['tweet_text'][i] = ' '.join(
        re.sub(
            "\\b\\w{1,3}\\b", 
            " ", TweetsDS['tweet_text'][i]
        ).split()
    ).lower()
# TweetsDS.tweet_text.str.replace(r'\b(\w{1,3})\b', '')
print("\n\n---------------------->> HEAD ( after removing <=3 chars word) <<----------------------\n\n")
print(TweetsDS['tweet_text'].head())



#--------------------------------------- A N A L Y Z I N G  T E X T  F O R  S E N T I M E N T

# ----------------------- TextBlob categorising as 'Positive', 'Negative' or 'Neutral'.

TweetsDS['sentiment'] = ' '
TweetsDS['polarity'] = None
for i,tweets in enumerate(TweetsDS.tweet_text) :
    if( i == 20 ):
        print("\n\n---------------------->> TEXTBLOB - 20th tuple <<----------------------\n\n")
        print( "ORIGINAL TEXT: \t" + tweets)
    blob = TextBlob(tweets)
    if( i == 20 ):
        print("TOKENISATION - WORDS:\t")
        print(blob.words)
        print("LEMMATIZATION (input-tokens):\t")
        lemmatizer = WordNetLemmatizer()
        for individualWord in blob.words:
            print(individualWord + "->" + lemmatizer.lemmatize(individualWord))
        print("POS - TAGS:\t")
        print(blob.tags)
        print("CHUNKING - NOUN PHRASES:\t")
        print(blob.noun_phrases)
        print("SENTIMENT:\t")
        print(blob.sentiment)
       
    #    print("TAGS:\t")
    #    print(blob.lemm)
    TweetsDS['polarity'][i] = blob.sentiment.polarity
    if blob.sentiment.polarity > 0 :
        TweetsDS['sentiment'][i] = 'positive'
    elif blob.sentiment.polarity < 0 :
        TweetsDS['sentiment'][i] = 'negative'
    else :
        TweetsDS['sentiment'][i] = 'neutral'



print("\n\n---------------------->> SENTIMENT & POLARITY <<----------------------\n\n")
print(TweetsDS.head())

# ----------------------- PRINTING COUNT PLOT
print("\n\n---------------------->> COUNT PLOT <<----------------------\n\n")
print(TweetsDS.sentiment.value_counts());
sns.countplot(x='sentiment', data = TweetsDS);

#----------------------- Sentiment Distribution
plt.figure(figsize=(10,6))
sns.distplot(TweetsDS['polarity'], bins=30)
plt.title('SENTIMENT DISTRIBUTION',size = 15)
plt.xlabel('Polarity',size = 15)
plt.ylabel('Frequency',size = 15)
print("\n\n---------------------->> SENTIMENT DISTRIBUTION <<----------------------\n\n")
plt.show();

#--------------------------------------- M O S T  F R E Q U E N T L Y  A P P E A R I N G  W O R D S
words = []
words = [word for i in TweetsDS.tweet_text for word in i.split()]
freq = Counter(words).most_common(30)
freq = pd.DataFrame(freq)
freq.columns = ['word', 'frequency']
print("\n\n---------------------->> FREQUENTLY APPEARING WORDS <<----------------------\n\n")
print(freq.head())

plt.figure(figsize = (10, 10))
sns.barplot(y="word", x="frequency",data=freq);


TweetsDS.to_csv('tweet.csv',index=False)