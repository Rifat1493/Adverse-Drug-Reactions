import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from nltk.corpus import wordnet

np.random.seed(3)

def tokenize_only(text):
    text=str(text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.match("[a-z]", token):#for topic model use len(token)<2
           #token=''.join([j for j in token if j in include])
           token= re.sub(r'[^\x00-\x7F]+','', token)
           filtered_tokens.append(token)
          
    final_data=" ".join([i for i in filtered_tokens])
    return final_data




df= pd.read_csv('./Annotated corpus/binary100+98.tsv', delimiter='\t',encoding='ISO-8859-1')


df1=pd.DataFrame()
df2=pd.DataFrame()
df.to_csv(path_or_buf='./Annotated corpus/binary100+98.tsv',sep='\t',encoding='utf-8')

df1= df1.drop(["Unnamed: 0","Annotated Text","Semantic Type","Drug","Country by Location","Country by Zone"],1)
df1= df1.drop("Lang",1)
df1["Class"]=df1.index
df1["Class"]=0

df1=df1.head(1400)
df = pd.concat([df1,df2])
con=df["Class"]==0
df2=df[con]
df2=df2.head(100)
df = df.sample(frac=1, random_state=42)
data=df["Drug"].tolist()




Y=df["Class"].tolist()
lemma = WordNetLemmatizer()
doc_complete=df["Text"]
fl=open("all_list.txt","r")
my_list=[]
my_list1=fl.readlines()
for i in my_list1:
    c=i.strip('\n')
    my_list.append(c)
stop =(stopwords.words('english'))
other_list=["rt","amp","codi","new","codein"]
stop=stop+(my_list)+other_list
stop=set(stop)

def clean(doc):
    doc2=str(doc)
    doc2=doc2.lower() 
    doc2=tokenize_only(doc2)
    stop_free = " ".join([i for i in doc2.split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word,"n") for word in stop_free.split())
    normalized1 = " ".join(lemma.lemmatize(word,"v") for word in normalized.split())
    normalized2 = " ".join(lemma.lemmatize(word,pos=wordnet.ADJ) for word in normalized1.split())

    return normalized2

doc_clean = [clean(doc).split() for doc in doc_complete]


list1=[]
for doc in doc_clean:
    list1=list1+doc
words = list(set(list1))

#words.append("ENDPAD")
word2idx = {w: i for i, w in enumerate(words)}

X=[]
for doc in doc_clean:
    temp=[]
    for j in doc:
        temp.append(word2idx[j])
        
    X.append(temp)
    
    
    
maxlen = max([len(s) for s in X])

# Need not to convert it to Np.array
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(maxlen=25, sequences=X, padding="post",value=925)

Y=np.array(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)



#creating word embeddings
from numpy import array
from numpy import asarray
from numpy import zeros

print(len(words))

vocab_size=len(words)+1
embeddings_index = dict()
f = open('glove.6B.50d.txt',encoding="utf-8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 50))
for word, i in word2idx.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector




"""my_mod=model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)
y_prob = model.predict(X_train) 
y_classes = y_prob.argmax(axis=-1)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_classes)"""


from keras.models import Sequential,Input,Model
from keras.layers import Dense
from keras.layers import Flatten,Dropout,LSTM,Activation,GRU,SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
               
# create the model CNN 12,1,15,14
model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))
#model.add(Embedding(926, 32, input_length=25))
model.add(Dropout(0.25))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())              

#create another model CRNN 12,12,1,15
model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#create RNN
model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))
model.add(GRU(300))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#create RCNN

model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))

model.add( SimpleRNN(300, activation="relu", return_sequences=True))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#create model CNNA

from keras.layers import Add
from keras.layers.core import Permute

from keras.models import *


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    
    a = Permute((2, 1))(inputs)
   # a = Reshape((input_dim, 40))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(25, activation='softmax')(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Add()([inputs, a_probs])
    return output_attention_mul

input = Input(shape=(25,))
model = Embedding(input_dim=vocab_size, output_dim=50, input_length=25)(input)
#model = Dropout(0.25)(model)
model = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(model)
attention_mul = attention_3d_block(model)
attention_mul = Flatten()(attention_mul)
output = Dense(1, activation='sigmoid')(attention_mul)
model = Model(input,output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#out=model.predict_classes(X_test)

y_prob = model.predict(X_test) 

y_pred = np.argmax(y_prob, axis=1)

check_list=y_prob.tolist()

new_list1=[]
for i in check_list:
    if i[0]>=.50:
        new_list1.append(1)
    
    else:
        new_list1.append(0)


y_pred=np.array(new_list1)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')


#visualize the confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 4})# font size
plt.show()




idx2word = {v: k for k, v in word2idx.items()}
print(idx2word[343])











import numpy as np
import matplotlib.pyplot as plt

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

rvals = [.71,.71,.57,.61,.72]
rects1 = ax.bar(ind, rvals, width, color='r')
pvals = [.68,.71,.66,.72,.65]
rects2 = ax.bar(ind+width,pvals, width, color='g')
fvals = [.69,.71,.61,.66,.68]
rects3 = ax.bar(ind+width*2, fvals, width, color='b')

#ax.yticks(np.arange(0, 1, .1))
ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('CNN', 'CRNN', 'RCNN','RNN','CNNA') )
#ax.legend( (rects1[0], rects2[0], rects3[0]), ('Recall','Precision', 'F1 score') )
ax.legend((rects1[0], rects2[0], rects3[0]), ('Recall','Precision', 'F1 score'),loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()















