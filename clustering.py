from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import pandas as pd
from gensim import corpora
from nltk.corpus import wordnet
lemma = WordNetLemmatizer()
df_search = pd.DataFrame()
df=pd.DataFrame()
df_temp = pd.read_csv('final_cluster2.tsv', delimiter='\t')
#df_temp=df_temp.head(300)
doc_complete=df_temp["Text1"]
temp_text=df_temp["Text"]
zone=df_temp['Country by Zone']
location=df_temp['Country by Location']
temp_link=df_temp['Link']

fl=open("all_list.txt","r")
my_list=[]
my_list1=fl.readlines()
for i in my_list1:
    c=i.strip('\n')
    my_list.append(c)
stop = (stopwords.words('english'))
other_list=["rt","amp","codi","new","codein"]
stop=stop+(my_list)+other_list
stop=set(stop)

def clean(doc):
    doc2=str(doc)
    doc2=doc2.lower() 

    stop_free = " ".join([i for i in doc2.split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word,"n") for word in stop_free.split())
    normalized1 = " ".join(lemma.lemmatize(word,"v") for word in normalized.split())
    normalized2 = " ".join(lemma.lemmatize(word,pos=wordnet.ADJ) for word in normalized1.split())

    return normalized2



doc_clean = [clean(doc).split() for doc in doc_complete]

#doc_clean = [clean(doc) for doc in doc_complete]

dictionary = corpora.Dictionary(doc_clean)
"""dictionary.filter_extremes(no_below=1, no_above=0.9, keep_n=10000, keep_tokens=None)
v=dictionary.items()
v=dictionary.values()"""


doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

Lda = gensim.models.ldamodel.LdaModel
model = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=20)
x=model.print_topics(num_topics=15, num_words=10)
c=1
for i in x:
    
    print("Topic %d :"%c)
    print(i)
    c+=1





model_lda=model[doc_term_matrix]
df1=pd.DataFrame()
count=0
for result in model_lda:
    temp=max(result,key=lambda item:item[1])[0]
    if(temp ==3):
        df1.loc[count,"Text"]=temp_text[count]
        df1.loc[count,"Country by zone"]=zone[count]
        df1.loc[count,"Country by Location"]=location[count]
        df1.loc[count,"Link"]=temp_link[count]
    count+=1

df1.to_csv(path_or_buf='sale_data1.tsv',sep='\t',encoding='utf-8')


from collections import Counter
  

total=0
for dic in doc_clean:
        wordcounts=Counter(dic)   
        l=wordcounts["discount"]   
        total+=l
                            
print(total)

b= dictionary.values()

b.tolist()
for n in b:
    if(n=="busi"):
        print("yes")
fl2=open("checkfile11.txt","a",encoding="utf-8")

for n in b:
    fl2.write(n+"\n")
fl2.close()

""" Another modeling for LDA"""
#ldamodel = Lda(doc_term_matrix, num_topics=15, id2word = dictionary, passes=20)
from gensim.models import CoherenceModel

coherence_values = []
model_list = []
for num_topics in range(10, 30, 5):
    model =Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=20)
    model_list.append(model)
    coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())
    
x = range(10, 30, 5)    
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

model =Lda(doc_term_matrix, num_topics=70, id2word = dictionary, passes=20)
    
coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')

print(coherencemodel.get_coherence())

from sklearn.externals import joblib
filename = 'main_model10.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk
model = joblib.load(filename)

"""TF-idf gensim"""
tfidf_model = gensim.models.TfidfModel(doc_term_matrix, id2word=dictionary)
#lda_model = gensim.models.ldamodel.LdaModel(tfidf_model[doc_term_matrix], id2word=dictionary, num_topics=20)
print((tfidf_model[doc_term_matrix]))
print(tfidf_model[doc_term_matrix].shape)

""" TF IDF SK learn"""

from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(doc_clean) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

films = { 'Text': doc_clean,'cluster': clusters}

frame = pd.DataFrame(films, index = [clusters] , columns = ['Text','cluster'])

terms = tfidf_vectorizer.get_feature_names()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = tfidf_vectorizer.get_feature_names()
for i in range(5):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end=',')
    print()
    
    
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(tfidf_matrix)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


cluster_range = range( 1, 30 )
cluster_errors = []
for num_clusters in cluster_range:
   clusters = KMeans( num_clusters )
   clusters.fit(tfidf_matrix )
   cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )