import pandas as pd
df1= pd.read_csv('./Annotated corpus/annotated_corpus.tsv', delimiter='\t',encoding='ISO-8859-1')
b=df1['Annotated Text']
search_list=b.tolist()

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

df=pd.DataFrame()
df["CUI"]=df.index
df["Annotated Text"]=df.index
df["MedDRA Term"]=df.index

"""for i in range(1,11):
    df["LLT Term "+ str(i)]=df.index
    df["LLT Code "+str(i)]=df.index
       
       
for i in range(1,4):
    df["PT Term "+ str(i)]=df.index
    df["PT Code "+str(i)]=df.index"""
       


##get at ticket granting ticket for the session
#search_list=['violent','kidney stone',"can't suffuse to blood"]
tgt = gettgt()
uri1 = "https://uts-ws.nlm.nih.gov/rest"
count=0
for word in search_list:

    st=getst(tgt)
    content_endpoint ="/search/current?string="+word+"&sabs=MDR&searchType=exact"#APPROXIMATE_MATCH"
   
    #content_endpoint ="/search/current?string="+word+"&sabs=MDR&searchType=approximate&PageNumber=1"#APPROXIMATE_MATCH"
    query = {'ticket':st}
    r = requests.get(uri1+content_endpoint,params=query)
    r.encoding = 'utf-8'
    items  = json.loads(r.text)
    #print(items)
    jsonData = items["result"]
    print(jsonData)
    
    for result in jsonData["results"]:
        
        #print(word+'\t',end="")
        df.loc[count,"CUI"]=result["ui"]
        df.loc[count,"Annotated Text"]=word
        df.loc[count,"MedDRA Term"]=result["name"]
        count+=1
    
       #print("ui: " + result["ui"]+'\t',end="")
       # print("name: " + result["name"]+'\t')
  
b1=df['CUI']
search_list1=b1.tolist()  

uri2 = "https://uts-ws.nlm.nih.gov"
count=0
for cui in search_list1:
    
    if(cui=="NONE"):
        count+=1
        continue
    else:
        content_endpoint = "/rest/content/"+'current'+"/CUI/"+str(cui)+'/atoms?sabs=MDR'
        
        ticket=getst(tgt)
        ##ticket is the only parameter needed for this call - paging does not come into play because we're only asking for one Json object
        query = {'ticket':ticket}
        r = requests.get(uri2+content_endpoint,params=query)
        print(r)
        r.encoding = 'utf-8'
        items  = json.loads(r.text)
        jsonData = items["result"]
        #print (jsonData)
        llt_count=0
        pt_count=0
        for result in jsonData:
            if result["termType"]=="LLT":
                 llt_count+=1
                 df.loc[count,"LLT Term "+ str(llt_count)]=result["name"]
                 df.loc[count,"LLT Code "+ str(llt_count)]=result["code"][-8:]
            else:
                pt_count+=1
                df.loc[count,"PT Term "+ str(llt_count)]=result["name"]
                df.loc[count,"PT Code "+ str(llt_count)]=result["code"][-8:]
                
    count+=1
            
            
df.to_csv(path_or_buf='mapping_annotated_data.tsv',sep='\t',encoding='utf-8')


con=df["CUI"]!="NONE"

df7=df[con]