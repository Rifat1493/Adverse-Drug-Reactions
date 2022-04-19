
#NEED TO CHANGE THE OUTPUT FILE

file = open("../metamap/file1.txt", "r")
final_data = file.readlines()

import requests
import lxml.html as lh
from lxml.html import fromstring
import json
import pandas as pd

uri="https://utslogin.nlm.nih.gov"
auth_endpoint = "/cas/v1/api-key"
apikey = '9d865463-19f3-46d1-99c1-8139d4a38f83'
service = "http://umlsks.nlm.nih.gov"
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
   
tgt = gettgt()

uri2 = "https://uts-ws.nlm.nih.gov"
main_df=pd.DataFrame()

length=len(final_data)
for count in range(0,length):
    
    
    cui=final_data[count]
    cui=cui.strip()
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
        main_df.loc[count,"CUI"]=cui
        for result in jsonData:
            if result["termType"] == "LLT" and llt_count == 0:
                 llt_count=1
                 main_df.loc[count,  "LLT Code"]=result["code"][-8:]
                 main_df.loc[count, "LLT Term"]=result["name"]
                 
                 
            elif result["termType"]=="LLT" and pt_count==0:
                pt_count=1
                main_df.loc[count, "PT Code"]=result["code"][-8:]
                main_df.loc[count, "PT Term"]=result["name"]
    except ValueError:
        continue
# need to change the ouput file
           
main_df.to_csv(path_or_buf='output1.tsv', sep='\t', encoding='utf-8')



