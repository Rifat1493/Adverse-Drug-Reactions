# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:17:40 2018

@author: Rifat
"""
from metaphone import doublemetaphone
import pandas as pd
file=open("DrugList.txt","r")
list1=file.readlines()
list1=["morphine"]
df=pd.DataFrame()

for f in list1:
    name_main = f.strip("\n")
    #name_main="fantanyl"
    """df[name_main]=df.index
    main_phonetic=doublemetaphone(name_main)
    count=0"""
    
    string1= f.strip("\n")
    
    alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
   
    for i in range(1,len(string1)):    #for deleting in 1 edit distance
        blist=list(string1)
        blist.pop(i)
        test = ''.join(blist)
        print(test)
        """test_phonetic=doublemetaphone(test)
        if test_phonetic[0].startswith(main_phonetic[0]) and string1!=test:
           df.loc[count,name_main]=test
           count=count+1"""
     
           
    for i in range (1,len(string1)):   #for replacing in 1 edit distance
        for j in alphabet:
            blist=list(string1)
            blist[i]=j
            
            test=''.join(blist)
            print(test)
            """test_phonetic=doublemetaphone(test)
            if test_phonetic[0].startswith(main_phonetic[0]) and string1!=test:
               df.loc[count,name_main]=test
               count=count+1"""
               
    for i in range (1,len(string1)):  #for insertion in 1 edit distance ?? previously it was len(string)+1
        for j in alphabet:
            test=string1[:i]+j+string1[i:] #character insertion in a fixed position
            print(test)
            """test_phonetic=doublemetaphone(test)
            if test_phonetic[0].startswith(main_phonetic[0]) and string1!=test:
               df.loc[count,name_main]=test
               count=count+1"""
               
               
df.to_csv(path_or_buf='DrugList_variants.csv',encoding='utf-8')

#Custom search engine API

from googleapiclient.discovery import build


def google_results_count(query):
    service = build("customsearch", "v1",
                    developerKey="AIzaSyAuBScRuFNqd02YeePVcY9NNIfZ6iYy0Xw")

    result = service.cse().list(
            q=query,
            cx='017167365924894062333:wmtl7zaiit0'
        ).execute()

    return result["searchInformation"]["totalResults"]    


for name in list1:

  df1=pd.DataFrame()
  df1[name.strip('\n')]=df1.index
  df1['count']=df1.index
  for i in range (len(df[name.strip('\n')])):
      if (type(df.loc[i,name.strip('\n')]))==str:
          df1.loc[i,name.strip('\n')]=df.loc[i,name.strip('\n')]
          df1.loc[i,'count']=int(google_results_count(df.loc[i,name.strip('\n')]))
          temp=df1.sort_values('count',ascending=[0])
          temp1=temp.head(5)
          for k in temp1[name.strip('\n')]:
              file2=open("DrugList_variants1.txt","a")
              file2.write(k+'\n')
    
    
h=google_results_count("fantanil drug")

print(h)


x=doublemetaphone("morphin")
print(x)