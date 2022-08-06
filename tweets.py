import twint
from datetime import date
import numpy as np
import pandas as pd
import model

# Here, let's define some constant variables
FOLLOWER_DIVISION = 50000000

# Let's also create a dictionary to calculate the overall sentiment of companies
company_value = dict()

# Let's create a class for each indivdiual tweet. 
# Let's use the persons follower count, likes on the tweet, and retweets all to help us find the weight of the tweet
class Tweet:
    def __init__(self, company, tweet, account, likes, retweets):
        self.tweet = tweet
        self.account = account
        self.company = company
        self.retweets = retweets+1
        self.likes = likes+1
    
    def get_tweet_score(self):
        tweet_score = model.sentiment(self.tweet) -0.5
        tweet_score *= self.likes + self.retweets*10
        return tweet_score

# Here we're using a numpy array as python doesn't support arrays out of the box, instead they support lists, which are much slower than arrays (I don't plan to edit any of these values)
# I may switch from a numpy array to a text file as I may make this web-based so people can input what companies they want to track
# For now, I'm keeping my company list short (for testing purposes), in deployment, I want to search all of twitter. 

companies = np.empty(7, dtype=object)
companies = ['twitter', 'tesla', 'apple', 'google', 'meta', 'netflix', 'amazon']

# For testing, to keep our program short, let's go through a small amount of twitter accounts
#accounts = np.empty(18, dtype=object)
#accounts = twitteraccounts = ['elonmusk', 'Stocktwits', 'SJosephBurns', 'PeterLBrandt', 'MarketWatch', 'steve_hanke', 'PeterSchiff', 'RedDogT3', 'allstarcharts', 'IBDinvestors',
# 'hmeisler', 'WSJmarkets', 'WarrenBuffett','BreakoutStocks', 'bespokeinvest', 'Stephanie_Link', 'nytimesbusiness', 'WSJDealJournal']

# Calculates the date a week ago
#a_week_ago = datetime.now() - timedelta(days=1)
a_week_ago = date.today()
a_week_ago = a_week_ago.strftime("%Y-%m-%d")

# Now let's define some functions to help us rank companies
def search_tweet(search):# acc):
    # Configure twint
    c = twint.Config()
    c.Lang = 'en'
    #c.Username = acc
    c.Hide_output = True
    c.Limit = 100
    c.Since = a_week_ago
    c.Pandas = True
    c.Search = search
    return_df = pd.DataFrame() 
    twint.run.Search(c)

    tweets_df = twint.storage.panda.Tweets_df
    return_df = pd.concat([return_df, tweets_df])

    return return_df



for company in companies:

    # Let's call search for tweets from containing references from each company
    company_score = 0
    unformattted_tweets_df = search_tweet(company)
    unformattted_tweets_df.to_csv('data.csv')
    object_values = [company]

    # Let's create a list with values needed to initilize each tweet as an object
    try:
        for tweet_text in unformattted_tweets_df['tweet']:
            object_values.append(tweet_text)
            break

        for tweet_acc in unformattted_tweets_df['username']:
            object_values.append(tweet_acc)
            break

        for tweet_likes in unformattted_tweets_df['nlikes']:
            object_values.append(tweet_likes)
            break

        for tweet_rt in unformattted_tweets_df['nretweets']:
            object_values.append(tweet_rt)
            break
        
        # Let's make our tweet an object  
        tweet_obj = Tweet(object_values[0], object_values[1], object_values[2], object_values[3], object_values[4])    
        company_score += tweet_obj.get_tweet_score() 

    except KeyError:
        print('EXEPTION')

    company_value[company] = company_score


# Now that we have all our company values. Let's check which ones are negitive, which are positive, etc.
very_negitive = []
negitive = []
neutral = []
positive = []
very_positive = []
for company in company_value:
    score = company_value[company].astype(float)
    if score <= 0:
        very_negitive.append(company)
    elif score < 0.1 and score > 0:
        negitive.append(company)
    elif score >= 0.1 and score <= 0.3:
        neutral.append(company)
    elif score > 0.3 and score <= 0.4:
        positive.append(company)
    else:
        very_positive.append(company)

# Let's print this, but, let's have the values look quite nice
print('\n ---- WORST Companies To Invest in----')
for very_negitive_company in very_negitive:
    print(very_negitive_company.capitalize())

print('\n ---- Do NOT Invest In These Companies ----')
for negitive_company in negitive:
    print(negitive_company.capitalize())

print('\n ---- These Companies Are A Tossup ----')
for neutral_company in neutral:
    print(neutral_company.capitalize())

print('\n ---- Invest In These Companies ----')
for positive_company in positive:
    print(positive_company.capitalize())

print('\n ---- BEST Companies To Invest In ----')
for very_positive_company in very_positive:
    print(very_positive_company.capitalize())


# Add a breather line of space
print('\n')
