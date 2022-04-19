from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains yours credentials to access Twitter API 
access_token = "798499970907111424-ZVpZxBWUgQYyYs0TExZLdqnCSNLdqJ2"
access_token_secret = "Ud2XEsSDTdkpd3BYNfFrNKIGFZagwNQ866ANaDBRPc3tG"
consumer_key = "lU5fgANUuk49Yr5KzhRj4X7EA"
consumer_secret = "MDjHgN5IHmRXbGfcUn02YzxcjXj6JxHaVXFHUYaGaRFKTmQ0iJ"

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    
    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l,tweet_mode='extended')

    fl=open("test.txt","r")
    my_list=fl.readlines()
    #This line filter tweets from the words.
    stream.filter(track=my_list)
       

