# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:02:15 2018

@author: Rifat"""


import json
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = '798499970907111424-ZVpZxBWUgQYyYs0TExZLdqnCSNLdqJ2'
ACCESS_SECRET = 'Ud2XEsSDTdkpd3BYNfFrNKIGFZagwNQ866ANaDBRPc3tG'
CONSUMER_KEY = 'lU5fgANUuk49Yr5KzhRj4X7EA'
CONSUMER_SECRET = 'MDjHgN5IHmRXbGfcUn02YzxcjXj6JxHaVXFHUYaGaRFKTmQ0iJ'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
# Initiate the connection to Twitter REST API
twitter = Twitter(auth=oauth)
            
# Search for latest tweets about "#nlproc"
#fl=open("variants_druglist.txt","r")
#my_list=fl.readlines()
#twitter.search.tweets(q=my_list)
tweets=twitter.search.tweets(q='narcotic', result_type='recent', lang='en', count=10)
print(type(tweets))
with open('data.txt', 'w') as outfile:
    
     json.dump(tweets, outfile,indent=2)
    
    
    
tweets_data_path = 'data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
    
print(len(tweets_data))


tweets_filename = 'data.txt'
tweets_file = open(tweets_filename, "r")

for line in tweets_file:
    try:
        # Read in one line of the file, convert it into a json object 
        tweet = json.loads(line.strip())
        if 'text' in tweet: # only messages contains 'text' field is a tweet
            print(tweet['id']) # This is the tweet's id
            print(tweet['created_at']) # when the tweet posted
           

            hashtags = []
            for hashtag in tweet['entities']['hashtags']:
            	hashtags.append(hashtag['text'])
            print (hashtags)

    except:
        # read in a line is not in JSON format (sometimes error occured)
        continue
fl = open("test11.txt","a")

print(tweets)

fl.close()

