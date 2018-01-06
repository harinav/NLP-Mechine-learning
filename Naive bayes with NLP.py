import nltk
from nltk.corpus import wordnet
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from  sklearn import svm
import string
from collections import Counter

train=raw_input("enter_train textfile path: ")
test=raw_input("enter _test textfile path: ")
short_train = open(train,"r").read()
short_test=open(test,"r").read()
documents = []
documents_test=[]
all_words=[]
all_test=[]
all_words_syn=[]

def clear_punctuation(s):
    clear_string = ""
    for symbol in s:
        if symbol not in string.punctuation:
            clear_string += symbol
	else :
	    clear_string +=" "
    return clear_string

short_test=clear_punctuation(short_test)

short_train=clear_punctuation(short_train)

stop_words = set(stopwords.words('english'))

for r in short_train.split('\n'):

    documents.append(r)

for t in short_test.split('\n'):

     documents_test.append(t)

short_pos_words=word_tokenize(short_train)

filtered_sentence = [w for w in short_pos_words if not w in stop_words]

filtered_sentence = []

for w in short_pos_words:
    if w not in stop_words:
        filtered_sentence.append(w)

short_test_words=word_tokenize(short_test)

filtered_sentence_test = [w for w in short_test_words if not w in stop_words]

filtered_sentence_test = []

stem =nltk.PorterStemmer()

for w in short_test_words:
    if w not in stop_words:
        filtered_sentence_test.append(w)
for w in filtered_sentence:
    s= stem.stem(w.lower())
    all_words.append(stem.stem(s))
for w in filtered_sentence_test:
    s=stem.stem(w.lower())
    all_test.append(stem.stem(s))
all_words = [str(x) for x in all_words]
all_test=[str(x)for x in all_test]
for sy in all_words:
	all_words_syn.append(sy)
	for syn in wordnet.synsets(sy):
  		  for l in syn.lemmas():
        	  	all_words_syn.append(stem.stem(l.name()))
all_words_syn=[str(x) for x in all_words_syn]
all_words_syn1=nltk.FreqDist(all_words_syn)
all_test1=nltk.FreqDist(all_test)
print all_words_syn1.keys()
print all_test1.keys()
cosine_doc=(
all_words_syn1.keys())
cosine_doc_target=(
all_words_syn1.keys())
text_clf = Pipeline([ ('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(cosine_doc,cosine_doc_target)
predicted = text_clf.predict(all_test1.keys())
print (np.mean(predicted ==all_test1.keys())*100)


