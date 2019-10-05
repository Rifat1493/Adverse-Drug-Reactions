# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:49:57 2018

@author: Rifat
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import nltk
include=set(string.printable)
tweets_data_path = 'D:\\Research\\Raw data\\drug_data4.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
print(len(tweets_data))
tweets = pd.DataFrame()
df=pd.DataFrame()
df1 = pd.DataFrame()
df3 = pd.DataFrame()

"""tweets['User ID'] = list(map(lambda tweet: tweet['id'], tweets_data))
tweets['Text ID'] = list(map(lambda tweet: tweet['id_str'], tweets_data))
tweets['child'] = map(lambda tweet: tweet.get('grandparent', {}).get('parent', {}).get('child') , tweets_data)
tweets['User ID'] = list(map(lambda tweet: tweet.get('id', {}),tweets_data))
tweets['Text ID'] = list(map(lambda tweet: tweet.get('id_str', {}),tweets_data))"""
tweets['Text'] = list(map(lambda tweet: tweet.get('extended_tweet', {}).get('full_text'),tweets_data))

tweets['Text1'] = list(map(lambda tweet: tweet.get('text', {}),tweets_data))
tweets['Lang'] = list(map(lambda tweet: tweet.get('lang', {}),tweets_data))
tweets['Country by Location'] = list(map(lambda tweet: tweet.get('user', {}).get('location'),tweets_data))
tweets['Country by Zone'] = list(map(lambda tweet: tweet.get('user', {}).get('time_zone'),tweets_data))

df=tweets
df = pd.concat([df,df1])

""" extendTweet merging"""

count=0
for i in range(98519):
    if(df.loc[count,"Text"])== None:
      df.loc[count,"Text2"]=df.loc[count,"Text1"]
    else:
      df.loc[count,"Text2"]=df.loc[count,"Text"]
    count+=1
    print(count)
    
    
df = df.drop('Text1', 1)

b=0
count=0
for i in range(68204):
    if(df.loc[count,"Text"])== None:
      b+=1
    
    count+=1
   
    
print((62460/68204)*100)

#Preprocessing TSV file
#df=pd.read_csv('Dcorpus1.tsv', delimiter='\t')
 
con=df["Lang"]=="en"

df=df[con]
#Removing Retweet

cond1=df['Text'].str.startswith(" RT")

cond_retweet=np.invert(cond1)

df=df[cond_retweet]

df3=df
df1=df
df2=df
#removing tweets with url
cond2=df['Text'].str.contains('https',na=False)
cond_url=np.invert(cond2)

df=df[cond_url]

#Copying retweets
con=df['Text'].str.startswith("RT")
df1=df[con]  #copy
cond_retweet=np.invert(con)
df=df[cond_retweet]
#remove retweet

df=df.drop_duplicates(subset='Text', keep='first', inplace=False)

df.to_csv(path_or_buf='./extended_data/preprocessed_binary_annotation.tsv',sep='\t',encoding='utf-8')

df = pd.read_csv('data_cluster.tsv', delimiter='\t',encoding='ISO-8859-1')
df= pd.read_csv('./extended_data/Raw_data.tsv', delimiter='\t',encoding='ISO-8859-1')


df3=pd.DataFrame()
#concate
df = pd.concat([df,df1])


"""Topic model Preprocess"""

def tokenize_only(text):
    text=str(text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.match("[a-zA-Z@#$0-9.,?/'%&():;]", token):#for topic model use len(token)<2
           #token=''.join([j for j in token if j in include])
           token= re.sub(r'[^\x00-\x7F]+','', token)
           filtered_tokens.append(token)
          
    #final_data=" ".join([i for i in filtered_tokens])
    final_data=""
    key=0
    for i in filtered_tokens:
        if i in ".,?/'():;"or i in "'s" or i in "n't":
            final_data=final_data+i
            
           
        elif (i in "@#$"):
            temp=i
            key=1
            
        else:
            if(key==1):
                final_data=final_data+" "+temp+i
                key=0
                
                
            else:
                final_data=final_data+" "+i
        
        
    
    return final_data


#imported data to df

df= pd.read_csv('sale_data1.tsv', delimiter='\t',encoding='ISO-8859-1')

#drop column
df = df.drop('Class', 1)




data=df['Text2']
data=data.tolist()

new_list=[]
for t in data:
    s=re.search('https[a-z0-9A-Z./:]{4,}',str(t))
    if(s==None):
        new_list.append("Null")
    else:
       b=s.group()
       new_list.append(b)
 
data = [re.sub('https[a-z0-9A-Z./:]{4,}', '', str(sent)) for sent in data]

data = [re.sub('[@#][a-z0-9A-Z_:-]{1,}',"",str(sent)) for sent in data]


data = [tokenize_only(doc) for doc in data]




df["Text"]=pd.Series(data)
df["Link"]=pd.Series(new_list)

df=df.drop_duplicates(subset='Text', keep='first', inplace=False)
df1.to_csv(path_or_buf='./Annotated corpus/mainannotated_corpus.tsv',sep='\t',encoding='utf-8')

#reoredering columns
df["Class"]=df.index
df["Class"]=0 


df = df[["Class",'Text', 'Country by Location','Country by Zone']]
 




fl2=open("tweetmetamap.txt","a",encoding="utf-8") 
for i in data:
   fl2.write(i+"\n")
   
   
   
   
   
"""PUBMED exploration"""

import pandas as pd
df= pd.read_csv('./pubmed/metamap12.tsv', delimiter='\t',encoding='ISO-8859-1')

final_data=[]

for i in range(1,47):
    data=df["CUI"+str(i)]
    data=data.tolist()
    data = [x for x in data if str(x) != 'nan']

    final_data=final_data+data

import requests
import lxml.html as lh
from lxml.html import fromstring
import json

uri="https://utslogin.nlm.nih.gov"
auth_endpoint = "/cas/v1/api-key"
apikey='9d865463-19f3-46d1-99c1-8139d4a38f83'
service="http://umlsks.nlm.nih.gov"
def gettgt():
 params = {'apikey': apikey}
 h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent":"python" }
 r = requests.post(uri+auth_endpoint,data=params,headers=h)
 response = fromstring(r.text)
 tgt = response.xpath('//form/@action')[0]
 return tgt

def getst(tgt): 
 params = {'service': service}
 h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent":"python" }
 r = requests.post(tgt,data=params,headers=h)
 st = r.text
 return st




"""df1["CUI"]=df.index

df1["LLT Code"]=df.index
df1["LLT Term"]=df.index
df1["PT Code"]=df.index
df1["PT Term"]=df.index"""
   
   
tgt = gettgt()

uri2 = "https://uts-ws.nlm.nih.gov"
df1=pd.DataFrame()
for count in range(0,57168):
    
    
    cui=final_data[count]
    content_endpoint = "/rest/content/"+'current'+"/CUI/"+cui+'/atoms?sabs=MDR'
   
    
    ticket=getst(tgt)
    
    ##ticket is the only parameter needed for this call - paging does not come into play because we're only asking for one Json object
    try:
        query = {'ticket':ticket}
        r = requests.get(uri2+content_endpoint,params=query)
        r.encoding = 'utf-8'
        items  = json.loads(r.text)
       
        jsonData = items["result"]
       
       # print (jsonData)
        llt_count=0
        pt_count=0
        df1.loc[count,"CUI"]=cui
        for result in jsonData:
            if result["termType"]=="LLT" and llt_count==0:
                 llt_count=1
                 df1.loc[count,"LLT Code"]=result["code"][-8:]
                 df1.loc[count,"LLT Term"]=result["name"]
                 
                 
            elif result["termType"]=="LLT" and pt_count==0:
                pt_count=1
                df1.loc[count,"PT Code"]=result["code"][-8:]
                df1.loc[count,"PT Term"]=result["name"]
               
                
        
    except ValueError:
        
        continue
          


#df3=pd.DataFrame()

df3 = pd.concat([df3,df1])
           
df3.to_csv(path_or_buf='umls_pubmed.tsv',sep='\t',encoding='utf-8')


con=df["CUI"]!="NONE"

df7=df[con]


print(final_data)