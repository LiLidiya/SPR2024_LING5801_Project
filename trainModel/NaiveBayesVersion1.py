import nltk
from nltk import ConditionalFreqDist
from nltk.tokenize import word_tokenize
import math
from collections import defaultdict
from operator import itemgetter
from tabulate import tabulate
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.utils import resample
import numpy as np

def trainNB(documents, classes):
    logprior = {}
    loglikelihood = {}
    vocab = set(word for _, doc in documents for word in doc)
    bigdoc = defaultdict(list)
    N_doc = len(documents)
    
    for c in classes:
        # Number of instances of class c in training corpus:
        N_c = sum(1 for doc_class, _ in documents if doc_class == c)
        # Calculate log prior:
        logprior[c] = math.log(N_c / N_doc)
        # Concatenate the documents for class c:
        bigdoc[c] = [word for doc_class, doc in documents if doc_class == c for word in doc]
        # Compute log likelihood for each word in vocabulary given class c:
        denom = sum(bigdoc[c].count(w) + 1 for w in vocab)

        for word in vocab:
            num = bigdoc[c].count(word) + 1
            loglikelihood[(word, c)] = math.log(num / denom)
    
    return logprior, loglikelihood, vocab

def testNB(testdoc, logprior, loglikelihood, classes, vocab):
    sums = {}
    for c in classes:
        sums[c] = logprior[c]
        for word in testdoc:
            if word in vocab:
                sums[c] += loglikelihood[(word,c)]
    return sorted(sums.items(), key=itemgetter(1), reverse = True)[0][0]


files = {"oneRate": open("oneRate500.txt","r"),
         "twoRate": open("twoRate500.txt","r"),
         "threeRate": open("threeRate500.txt","r"),
         "fourRate": open("fourRate500.txt","r"),
         "fiveRate": open("fiveRate500.txt","r")
         }


docs =[]
for fd in files:
    for line in files[fd].readlines():
        docs.append((fd,word_tokenize(line.lower())))

classes = ['oneRate','twoRate','threeRate','fourRate','fiveRate']

print("----------------------Start training--------------------------------")
priors, likelihood, vocab = trainNB(docs, classes)
print("----------------------Finished training--------------------------------")


testdata = open("testdata1rate.txt","r")

correct = 0
incorrect = 0

for i in testdata.readlines():
    test = i.lower().split()
    label = testNB(test, priors, likelihood, classes, vocab)

    if label == "oneRate":
        correct += 1
    else:
        incorrect += 1
print("Correct Assumption: {}\nIncorrect Assumption: {}".format(correct,incorrect))
### Based on 660 one rating test data. We trained our classifier based on 2500 reviews (500 reviews per rating). Need far more accuracy.
### Correct Assumption: 219
### Incorrect Assumption: 441


testdata.close()

for fd in files:
    files[fd].close()