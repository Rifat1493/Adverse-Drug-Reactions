# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:56:46 2018

@author: Rifat
"""
import pandas as pd
df = pd.read_csv('Variant_list//DrugList_variants.csv')

from googleapiclient.discovery import build


def google_results_count(query):
    service = build("customsearch", "v1",
                    developerKey="AIzaSyAuBScRuFNqd02YeePVcY9NNIfZ6iYy0Xw")

    result = service.cse().list(
            q=query,
            cx='017167365924894062333:wmtl7zaiit0'
        ).execute()

    return result["searchInformation"]["totalResults"] 

df1=pd.DataFrame()
df1['oxycontin']=df1.index
df1['count']=df1.index
for i in range (139):
    df1.loc[i,'oxycontin']=df.loc[i,'oxycontin']
    df1.loc[i,'count']=int(google_results_count(df.loc[i,'oxycontin']+" drug"))
    temp=df1.sort_values('count',ascending=[0])
    

temp.to_csv(path_or_buf='Variant_list//oxycontin.csv',encoding='utf-8')

print(google_results_count('prozc drug'))

""" Clustering Practice """


"""
Created on Wed Apr  4 16:46:15 2018

@author: Rifat
"""

#clustering by Biterm model


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import nltk
import pandas as pd
import re
from autocorrect import spell
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
lemma = WordNetLemmatizer()
def tokenize_only(text):
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
       # stems = [lemma.lemmatize(t) for t in filtered_tokens]
       #return stems
        return filtered_tokens


df1 = pd.DataFrame()
df = pd.read_csv('clustering data.tsv', delimiter='\t')
df1=df.head(150)
doc_complete=df["Text"]

fl=open("all_list.txt","r")
my_list=[]
my_list1=fl.readlines()
for i in my_list1:
    c=i.strip('\n')
    my_list.append(c)

stop = (stopwords.words('english'))
other_list=["rt","amp","https"]

stop=stop+(my_list)+other_list
stop=set(stop)
exclude = set(string.punctuation) 
include=set(string.printable)

def clean(doc):
    doc=str(doc)
    doc2=doc.lower()
    #doc2=" ".join(spell(word1) for word1 in doc1.split())
    
    #doc2=tokenize_only(doc2)
    #doc2=" ".join(doc2)
    stop_free = " ".join([i for i in doc2.split() if i not in stop])
    ascii_free=''.join([j for j in stop_free if j in include])
    punc_free = ''.join(ch for ch in ascii_free if ch not in exclude)
   # normalized = " ".join(lemma.lemmatize(word) for word in doc.split())
   
    
   
    return punc_free




doc_clean = [clean(doc).split() for doc in doc_complete]

test=doc_clean

print(stop)

from gensim import corpora
from gensim.corpora import Dictionary
dictionary = Dictionary(doc_clean)

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(doc_clean)
dictionary.filter_extremes(no_below=60, no_above=0.9, keep_n=10000, keep_tokens=None)
v=dictionary.items()

list1=[]


print(type(dictionary))
print(type(v))
for k in v:
    if(len(k[1])<3):
      list1.append(k[0])


dictionary.filter_tokens(bad_ids=list1)
    
b= dictionary.items()

for n in b:
    print(n)


# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]



didhum=doc_term_matrix.toarray()

from gensim.models import TfidfModel
from gensim.models import Word2Vec
modelvec = Word2Vec(doc_clean, min_count=1)
print(modelvec.shape)

modelvec1=modelvec.tolist()
vt=modelvec[doc_clean[0]]

model = TfidfModel(doc_term_matrix)  # fit model
vector = model[doc_term_matrix[2]]

print(type(model))
print(doc_term_matrix.shape)
prev_docClean=doc_clean

Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=20)
x=ldamodel.print_topics(num_topics=10, num_words=8)
c=1
for i in x:
    
    print("Topic %d :"%c)
    print(i)
    c+=1
    
    
from gensim.models.coherencemodel import CoherenceModel    
cm = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, coherence='u_mass')  # tm is the trained topic model
cm.get_coherence()

#similarity check
text = "does anyone have any morphine i can borrow once a month every month until I'm in my mid-to-late forties"

text=clean(text)
text=" ".join(text)
clean_text1= clean(text)
clean_text=clean_text1.split()
bow = dictionary.doc2bow(clean_text)
print(ldamodel[bow])
from gensim import similarities
 
lda_index = similarities.MatrixSimilarity(ldamodel[doc_term_matrix])
 
# Let's perform some queries
similarities = lda_index[ldamodel[bow]]
# Sort the similarities
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
 
# Top most similar documents:
print(similarities[:10])
# [(104, 0.87591344), (178, 0.86124849), (31, 0.8604598), (77, 0.84932965), (85, 0.84843522), (135, 0.84421808), (215, 0.84184396), (353, 0.84038532), (254, 0.83498049), (13, 0.82832891)]
 
# Let's see what's the most similar document
document_id, similarity = similarities[1]
print(document_id)
print(doc_complete[29884])

x=ldamodel[bow]  

"""Others methods"""
NUM_TOPICS=10
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model.fit_transform(doc_term_matrix)
print(lda_Z)
print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
 
print("LDA Model:")
print_topics(lda_model, vectorizer)
print("=" * 20)

from sklearn.feature_extraction.text import CountVectorizer
 
NUM_TOPICS = 10
 
vectorizer = CountVectorizer(min_df=.01, max_df=.90, 
                             stop_words='english', lowercase=True, 
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')


data_vectorized = vectorizer.fit_transform(df["Text"].values.astype('U')).toarray()
vectorizer.get_feature_names()
print((data_vectorized))

""" Finding the most business asssociated value"""


model_analysis=ldamodel[doc_term_matrix]
print(type(model_analysis[0]))
for j in model_analysis[0]:
    print((j[1]))
    
max(model_analysis[0],key=lambda item:item[1])[0]

df1=pd.DataFrame()
count=0

for result in model_analysis:
    temp=max(result,key=lambda item:item[1])[0]
    if(temp ==2):
        df1.loc[count,"Text"]=doc_complete[count]
        
    count+=1


df1=df1.drop_duplicates(subset='Text', keep='first', inplace=False)
df1.to_csv(path_or_buf='sample_classification.tsv',sep='\t',encoding='utf-8')


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

from collections import Counter
wordcounts = Counter(doc_clean[0])
wordcounts['lbc']   

total=0
for dic in doc_clean:
        wordcounts=Counter(dic)   
        l=wordcounts["best"]   
        total+=l
                            
print(total)



myDict = {'a':1,'b':2,'c':3,'d':4}

print(type(myDict))
if 'a' in myDict: 
    del myDict['a']
print(myDict)

doc2="RT @soar_institute: Exciting coverage form @baltimoresun about @onit4you's important work sharing #fentanyl test strips! #HarmReduction #We…"

doc2=doc2.lower()
print(doc2)
stop_free = " ".join([i for i in doc2.split() if i not in stop])
print(stop_free)
ascii_free=''.join([j for j in stop_free if j in include])
print(ascii_free)
punc_free =''.join(ch for ch in ascii_free if ch not in exclude)
print(punc_free)
new_list=[]
for value in stop:
    punc_free =''.join(ch for ch in value if ch not in exclude)
    new_list.append(punc_free)

b= dictionary.values()

b.tolist()
for n in b:
    if(n=="busi"):
        print("yes")
fl2=open("checkfile3.txt","a",encoding="utf-8")


for n in b:
    fl2.write(n+"\n")
fl2.close()


import string
include=set(string.printable)
file1=open("read_text.txt","r")
b=file1.readlines()
print(type(b))
n=len(b)
print(n)
for i in b:
    ascii_free=''.join([j for j in i if j in include])
    print(ascii_free)

c="inpbusinessmo"
if("business" in c):
    c="business"
print(c)

# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:01:15 2018

@author: Rifat
"""

#from lxml import etree
import xml.etree.ElementTree as ET
import pandas as pd

#xml_data = open('C:\\Users\\Rifat\\Desktop\\python\\FDA.xml').read()
#root= etree.iterparse('C:\\Users\\Rifat\\Desktop\\python\\FDA.xml', events=('start', 'end'))
#root = ET.XML(xml_data) # element tree
#root = ET.fromstring(xml_data)



tree = ET.parse('D:\\Research\\Raw data\\pubmed_result.xml')
root = tree.getroot()
all_records = []
for i, child in enumerate(root):
   record = {}
   for subchild in child:
       record[subchild.tag] = subchild.text
   all_records.append(record)
df1=pd.DataFrame(all_records)
            
df1.head(10)

import xml.etree.cElementTree as etree

# read in all PMIDs,Abstract title and Abstract Texts
pmid_abstract = []
file_path= "test_result.xml"
b=0
count=0
for event, element in etree.iterparse(file_path):
     
     if element.tag in ["sup"]:
         
         print(element.text)
         count+=1
         if(count==10):
           break
        
b=0
for event, element in etree.iterparse(file_path):
     print(element.tag)
     if(b==10):
       break
     b+=1
     
import xml.etree.ElementTree as ET

tree = ET.parse('test_result.xml')
root = tree.getroot()
print(root[2])
print(type(root[0].attrib))

count=0
for child in root.iter():
    
    if child.tag in ["PMID"]:
       print(child.text)
    
    if child.tag in ["AbstractText"]:
       b="".join(child.itertext())
       print(b)
       count+=1
    
    if(count==20):
        break
    
import xml.etree.ElementTree as ET

tree = ET.parse('test_result.xml')
root = tree.getroot()  

count=0
for child in root.iter():
    

    
    if child.tag in ["ELocationID"]:
      
       print((child.items()))
       child.clear()
       count+=1
    
    if(count==20):
        
        break
    

fl2=open("file34.txt","a")
fl2.write("hello")
fl2.write("world")

fl2.close()
print(len(b))




import xml.etree.ElementTree as ET

tree = ET.parse('test_result.xml')
root = tree.getroot()
my_text=""
count=0
check=0
for child in root.iter():
    
    if child.tag in ["PMID"]:
       check=1
       my_text=child.text
    
    if child.tag in ["AbstractText"]:
       b="".join(child.itertext())
       
       #re.sub(r'[^\x00-\x7F]+',' ', b)
       b1=''.join([j for j in b if j in include])
       
       
       
       
       if(count==30):
           print()
           break
       
       if(check==1):
         print("\n\n"+my_text,end="")
         print("\n"+b1,end="")
         check=0
         count+=1
       else:
         print(b1,end="")
         
""" Regex practice """

import re
text="cheap think itâs great idea"
reg=re.compile('\d\d+')
re.findall('[^\@]',text)
b=re.findall('https[a-z0-9A-Z./:]{4,1}', text)
re.sub('https[a-z0-9A-Z./:]{4,}','', text)
s=re.search('[a-z]', text)
b=s.group()
print((b))

reg.findall(text)
""" if("sale" in token):
    token="sale"
elif("business" in token):
    token="business"
elif("buy" in token):
    token="buy"
elif("price" in token):
    token="price"
elif("discount" in token):
    token="discount"
elif("cheap" in token):
    token="cheap"""
    
def remove_non_ascii_2(text):

    #return re.sub(r'[^\x00-\x7F]+','', text)  
    token=''.join([j for j in text if j in include])
    return token

import time
start_time = time.time()

doc_clean1 = remove_non_ascii_2('jitbo 😀😀😀 jit')
print(doc_clean1)
print("--- %s seconds ---" % (time.time() - start_time))



