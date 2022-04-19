import xml.etree.ElementTree as ET
#import re


import string

include=set(string.printable)
my_text=""
check=0
file_path= "D:\\Research\\Raw data\\pubmed_result.xml"
count=0
for event, child in ET.iterparse(file_path):
    
    if child.tag in ["PMID"]:
       check=1
       my_text=child.text
    
    if child.tag in ["AbstractText"]:
       b="".join(child.itertext())
      
       #re.sub(r'[^\x00-\x7F]+',' ', b)
       b1=''.join([j for j in b if j in include])
       
       
       
       if(count>=0 and count<9000):
           fl2=open("file1.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=9000 and count<18000):
           fl2=open("file2.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=18000 and count<27000):
           fl2=open("file3.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=27000 and count<36000):
           fl2=open("file4.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=36000 and count<45000):
           fl2=open("file5.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=45000 and count<54000):
           fl2=open("file6.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=54000 and count<63000):
           fl2=open("file7.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=63000 and count<72000):
           fl2=open("file8.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=72000 and count<81000):
           fl2=open("file9.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=81000 and count<90000):
           fl2=open("file10.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>=90000 and count<99000):
           fl2=open("file11.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
           
       elif(count>=99000 and count<=108000):
           fl2=open("file12.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()
       elif(count>108000):
           fl2=open("file13.txt","a",encoding="utf-8")
           if(check==1):
               fl2.write("\n\n"+my_text)
               fl2.write("\n"+b1)
               check=0
               count+=1
           else:
               fl2.write(b1)
           fl2.close()          
    
   child.clear()
 
       
""" Text handling of returned pubmed data"""

import pandas as pd
df=pd.DataFrame()

file_path= "texttweetmetamap.out"
#file_path= "read_text.xml"
count=-1
col=0

for event, child in ET.iterparse(file_path):
    
    if child.tag in ["UttText"]:
        ut_text=child.text
        ck_pmid=ut_text.split(' ', 1)[0]
    
    if child.tag in ["Token"]:
       pmid=child.text
       if(pmid.isdigit()):
           if(pmid==ck_pmid and len(pmid)>3):
              count+=1
              df.loc[count,"PMID"]=pmid
              
              col=0
       
    if child.tag in ["CandidateCUI"]:
       #print(child.text)
       temp_cui=child.text   
       chk=0
    if child.tag in ["CandidatePreferred"]:
       #print(child.text)
       temp_term=child.text
       
    if child.tag in ["SemType"]:
        if child.text in ["inpo","patf","comd","dsyn","emod","fndg","mobd","neop","sosy","menp"]:
            if(chk==0):
                chk=1
                col+=1
                df.loc[count,"CUI"+ str(col)]=temp_cui
                df.loc[count,"PTerm"+str(col)]=temp_term
                       
    child.clear()
    
df.to_csv(path_or_buf='./Annotated corpus/tweetmetamap_corpus.tsv',sep='\t',encoding='utf-8')
                
        
