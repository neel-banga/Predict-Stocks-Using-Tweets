import twint
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import model

# Here we're using a numpy array as python doesn't support arrays out of the box, instead they support lists, which are much slower than arrays (I don't plan to edit any of these values)
# I may switch from a numpy array to a text file as I may make this web-based so people can input what companies they want to track
# For now, I'm keeping my company list short (for testing purposes), in deployment, I want to search all of twitter. 

companies = np.empty(7, dtype=object)
companies = ['twitter', 'tesla', 'apple', 'google', 'meta', 'netflix', 'amazon']

# For testing, to keep our program short, let's go through a small amount of twitter accounts
accounts = np.empty(18, dtype=object)
accounts = twitteraccounts = ['elonmusk', 'Stocktwits', 'SJosephBurns', 'PeterLBrandt', 'MarketWatch', 'steve_hanke', 'PeterSchiff', 'RedDogT3', 'allstarcharts', 'IBDinvestors',
 'hmeisler', 'WSJmarkets', 'WarrenBuffett','BreakoutStocks', 'bespokeinvest', 'Stephanie_Link', 'nytimesbusiness', 'WSJDealJournal']

# Let's also create a dictionary to calculate the overall sentiment of companies
company_value = dict()

# Calculates the date a week ago
a_week_ago = datetime.now() - timedelta(days=7)
a_week_ago = a_week_ago.strftime("%Y-%m-%d")

# Now let's define some functions to help us rank companies
def search_tweet(search, acc):
    # Configure twint
    c = twint.Config()
    c.Lang = 'en'
    c.Username = acc
    c.Hide_output = True
    c.Since = a_week_ago
    c.Pandas = True
    c.Search = search
    return_df = pd.DataFrame() 
    twint.run.Search(c)

    tweets_df = twint.storage.panda.Tweets_df
    return_df = pd.concat([return_df, tweets_df])

    return return_df

for company in companies:
    company_score = 0
    for acc in accounts:
        unformattted_tweets_df = search_tweet(company, acc)

        # Here let's preform sentiment analysys and find our scores
        try :
            for tweet in unformattted_tweets_df['tweet']:
                company_score += model.sentiment(tweet)-0.5

        except KeyError:
            pass

    company_value[company] = company_score

# Now that we have all our company values. Let's check which ones are negitive, which are positive, etc.

negitive = []
neutral = []
positive = []
for company in company_value:
    score = company_value[company].astype(float)

    if score <= 0.1:
        negitive.append(company)
    elif score >= 0.1 and score <= 0.3:
        neutral.append(company)
    elif score > 0.3:
        positive.append(company)


# Create a getter for our main.py
def get_values():
    return negitive, neutral, positive

    
    
    
