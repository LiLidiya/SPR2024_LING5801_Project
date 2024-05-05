import nltk
from nltk import ConditionalFreqDist
from nltk.tokenize import word_tokenize
import math
import re
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

def imp(sentence, bigrams, fds, sumitems):
    initialProb = []
    indexList = []
    for key in sumitems:
        initialProb.append(sumitems[key])
    
    for i in range(5):
        indexList.append(i)
    
    problist = []
    for i in indexList:
        problist.append([i,sumitems[fds[i]]])
        
    for i in range(len(sentence)-1):
        for j in range(len(problist)):
            freq = bigrams[problist[j][0]][sentence[i]].freq(sentence[i+1])
            if freq > 0:
                problist[j][1] += math.log(freq) * 0.05     
            else:
                problist[j][1] += math.log(0.00001) * 0.05   # Give penalty
    
    maxdiff = -9999  # Initialize maxdiff with positive infinity
    max_label = None

    for i in problist:
        if i[1] > maxdiff:
            maxdiff = i[1]
            max_label = fds[i[0]]

    return max_label
    

def trainNB(documents, classes):
    logprior = {}
    loglikelihood = {}
    vocab = set(word for _, doc in documents for word in doc)       # All words from 5 txt files without repetition
    bigdoc = defaultdict(list)
    N_doc = len(documents)
    
    for c in classes:
        # Number of instances of class c in training corpus:
        N_c = sum(1 for doc_class, _ in documents if doc_class == c)
        # Calculate log prior:
        logprior[c] = math.log(N_c / N_doc)
        # Concatenate the documents for class c:
        bigdoc[c] = [word for doc_class, doc in documents if doc_class == c for word in doc]    # Store all words occured in the doc c
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
    return sorted(sums.items(), key=itemgetter(1), reverse = True)[0][0],sums

files = {"oneRate": open("TrainingDataV2/oneRate.txt","r"),
         "twoRate": open("TrainingDataV2/twoRate.txt","r"),
         "threeRate": open("TrainingDataV2/threeRate.txt","r"),
         "fourRate": open("TrainingDataV2/fourRate.txt","r"),
         "fiveRate": open("TrainingDataV2/fiveRate.txt","r")
         }


docs =[]
for fd in files:
    for line in files[fd].readlines():
        line = line.lower()
        docs.append((fd,word_tokenize(line)))

unigrams = []
bigrams = []
trigrams = []
for i in range(5):
    unigrams.append(nltk.FreqDist())
    bigrams.append(nltk.ConditionalFreqDist())
    trigrams.append(nltk.ConditionalFreqDist())

fds = ["oneRate","twoRate","threeRate","fourRate","fiveRate"]

for sentence in docs:
    cur = fds.index(sentence[0])
    sentence = sentence[1]
    for word in sentence:
        unigrams[cur][word] += 1

for sentence in docs:
    cur = fds.index(sentence[0])
    sentence = sentence[1]
    preceding = "<s>"
    for word in sentence:
        bigrams[cur][preceding][word] += 1
        preceding = word
    bigrams[cur][preceding]["</s>"] += 1

for sentence in docs:
    cur = fds.index(sentence[0])
    sentence = sentence[1]
    cond1 = "<s>"
    cond2 = "<s>"
    for word in sentence:
        context = " ".join([cond1,cond2])
        trigrams[cur][context][word] += 1
        cond1 = cond2
        cond2 = word

classes = ['oneRate','twoRate','threeRate','fourRate','fiveRate']

print("----------------------Start training--------------------------------")
priors, likelihood, vocab = trainNB(docs, classes)
print("----------------------Finished training--------------------------------")


testdata ={"oneRate": open("1.txt","r"), 
            "twoRate":open("2.txt","r"),
           "threeRate":open("3.txt","r"), 
           "fourRate":open("4.txt","r"), 
           "fiveRate":open("5.txt","r")} 

overallAccuracy = 0
changed = 0

for fd in testdata:
    correct,incorrect,one,two,three,four,five = 0,0,0,0,0,0,0
    for line in testdata[fd].readlines():
        test = word_tokenize(line.lower())
        label,sumItem = testNB(test, priors, likelihood, classes, vocab)

        prev = label 
        label = imp(test, bigrams,trigrams,fds,sumItem)
        cur = label 
        if label == fd:
            correct += 1
        else:
            incorrect += 1
            if label == "oneRate": one += 1
            elif label == "twoRate": two += 1
            elif label == "threeRate": three += 1
            elif label == "fourRate": four += 1
            else: five += 1
    overallAccuracy += correct/(correct+incorrect)
    print("Analysis for {}.txt".format(fd))
    print("Correct Assumption: {}\nIncorrect Assumption: {}".format(correct,incorrect))
    print("Accuracy: {}".format(correct/(correct+incorrect)))
    print("one:{} two:{} three:{} four:{} five:{}".format(one,two,three,four,five))
    print()

print("Overall accuracy of the classifier: {}".format(overallAccuracy/len(testdata.keys())))

for fd in testdata:
    testdata[fd].close()

for fd in files:
    files[fd].close()


