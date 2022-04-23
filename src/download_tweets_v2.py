from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


dictionary = {}
with open("input/key.txt") as f:
    for line in f:
       (key, val) = line.split(" ")
       dictionary[key] = val.strip()
       #print(line.split(" "))

#Variables that contains yours credentials to access Twitter API 
access_token = dictionary['access_token']
access_token_secret = dictionary['access_token_secret']
consumer_key = dictionary['consumer_key']
consumer_secret = dictionary['consumer_secret']







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
       

