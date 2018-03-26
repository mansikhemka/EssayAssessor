import pandas as pd
import numpy as np
import nltk
import re
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

df = pd.read_excel('/Users/mansikhemka/Desktop/training_set_rel3.xls')
df.fillna(0, inplace = True)
df = df[['essay_id', 'essay_set','essay','domain1_score']]
df.head(3)

list_essays={}
index = 0
for i in range(1,8):
    list_essay = []
    while (df.iloc[index]['essay_set'] == i):
        essay = df.iloc[index]['essay']
        essay = re.split('[?.,!]', essay)
        list_sentences = []
        for j in range(len(essay)):
            list_sentences.append(nltk.word_tokenize(essay[j]))
#         list_inner.append(np.array(nltk.word_tokenize(df.iloc[index]['essay'])))
        list_essay.append(list_sentences)
        index = index+1
    list_essays["string{0}".format(i)] = list_essay
print(list_essays['string2'][0]) 

essays = {}
for i in range(1,8):
    essay_set = list_essays["string{0}".format(i)]
    essay_vec = []
    for j in range(len(essay_set)):
        sentences = essay_set[j]
        sen = []
        senetence = []
        for k in range(len(sentences)-1):
            sentence=[sentences[k]]
            if(len(sentence[0])>0):
                model = Word2Vec(sentence, min_count=1)
                vectors = model[model.wv.vocab]
                sen.append(vectors)
                sentence = []
                
        essay_vec.append(sen)
    essays["string{0}".format(i)] = essay_vec
    
    
    
