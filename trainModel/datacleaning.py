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

def imp(sentence, label, bigrams, fds, sumitems):
    print("{} and index {}".format(label,fds.index(label)))   #####################################
    
    for i in sumitems:
        print(sumitems[i],end=" ")
    print("----------------------------------------------------")

    initialProb = []
    indexList = []
    for key in sumitems:
        initialProb.append(sumitems[key])
    
    if label == "oneRate":      # We only consider oneRate and twoRate
        indexList.extend([0,1])
    elif label == "fiveRate":   # We only consider fourRate and fiveRate
        indexList.extend([3,4])
    else:
        i = fds.index(label)    # We consider preceding label, current label, and next label
        indexList.extend([i-1,i,i+1])
    
    for i in indexList:     
        if sumitems[label] - sumitems[fds[i]] > 3:
            indexList.remove(i)     # If difference in initial guess is bigger than 1.2, remove it. We are not gonna consider such index
    
    if len(indexList) == 1:
        # print("returning without any change to label")
        return label    # length 1 means there is no other labels to consider except itself

    problist = []
    for i in indexList:
        problist.append([i,sumitems[fds[i]]])
        
    for i in range(len(sentence)-1):
        # print(sentence[i], sentence[i+1])
        for j in range(len(problist)):
            freq = bigrams[problist[j][0]][sentence[i]].freq(sentence[i+1])
            if freq > 0:
                problist[j][1] += math.log(freq) * -0.75     # -0.05
                # print("{} freq and {} log".format(freq,math.log(freq)*-0.05))
            else:
                problist[j][1] += math.log(0.0005) * -0.75   # Give penalty
    
    # for i in problist:  ###########################
    #     print("{}:{}".format(i[0],i[1]),end=" ")
    # print()
    

    maxdiff = 2000  # Initialize maxdiff with negative infinity
    max_label = None

    for i in problist:
        # diff = abs(sumitems[fds[i[0]]] - i[1])
        diff = i[1] - sumitems[fds[i[0]]]
        if diff < maxdiff:
            maxdiff = diff
            max_label = fds[i[0]]

    print("return {}".format(max_label))   #############################################
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
        print("finished one")
    
    return logprior, loglikelihood, vocab

def testNB(testdoc, logprior, loglikelihood, classes, vocab):
    sums = {}
    for c in classes:
        sums[c] = logprior[c]
        for word in testdoc:
            if word in vocab:
                sums[c] += loglikelihood[(word,c)]
    # print(sums.items())
    return sorted(sums.items(), key=itemgetter(1), reverse = True)[0][0],sums


files = {"oneRate": open("TrainingDataV2/oneRate.txt","r"),
         "twoRate": open("TrainingDataV2/twoRate.txt","r"),
         "threeRate": open("TrainingDataV2/threeRate.txt","r"),
         "fourRate": open("TrainingDataV2/fourRate.txt","r"),
         "fiveRate": open("TrainingDataV2/fiveRate.txt","r")
         }


docs =[]
# #### With data cleaning
for fd in files:
    for line in files[fd].readlines():
        line = line.lower()
        # line = re.sub(r',','',line)   # remove dot and comma
        # line = re.sub(r'\bhe\b|\bshe\b|\bhim\b|\bher\b','',line)   # remove irrelavent words
        line = re.sub(r'\bthe\b|\bto\b|\band\b','',line) # (296,364)
        # line = re.sub(r'[a-z]+[0-9]+','',line)   # like bio2013
        # line = re.sub(r'no comments','',line)   # there were some "no comments" in reviews
        docs.append((fd,word_tokenize(line)))



### For bigram FreqDist
unigrams = []
for i in range(5):
    unigrams.append(nltk.FreqDist())

bigrams = []
for i in range(5):
    bigrams.append(nltk.ConditionalFreqDist())

    
fds = ["oneRate","twoRate","threeRate","fourRate","fiveRate"]

for sentence in docs:
    cur = fds.index(sentence[0])
    for word in sentence[1]:
        unigrams[cur][word] += 1

for sentence in docs:
    cur = fds.index(sentence[0])
    sentence = sentence[1]
    preceding = "<s>"
    for word in sentence:
        bigrams[cur][preceding][word] += 1
        preceding = word
    bigrams[0][preceding]["</s>"] += 1
    cur += 1

classes = ['oneRate','twoRate','threeRate','fourRate','fiveRate']

print("----------------------Start training--------------------------------")
priors, likelihood, vocab = trainNB(docs, classes)
print("----------------------Finished training--------------------------------")


testdata ={"oneRate": open("1.txt","r"), # TestData/testdata1rate.txt
            "twoRate":open("2.txt","r"),
           "threeRate":open("3.txt","r"), # TestData/testdata3rate.txt
           "fourRate":open("4.txt","r"), # TestData/testdata4rate.txt
           "fiveRate":open("5.txt","r")} # TestData/testdata5rate.txt 

overallAccuracy = 0
changed = 0
for fd in testdata:
    correct,incorrect,one,two,three,four,five = 0,0,0,0,0,0,0
    for line in testdata[fd].readlines():
        test = word_tokenize(line.lower())
        label,sumItem = testNB(test, priors, likelihood, classes, vocab)

        prev = label #############################################
        if fd not in "oneRate fiveRate":
            label = imp(test,label,bigrams,fds,sumItem)
            cur = label #############################################
            if prev != cur: #################################
                changed += 1    ########################################

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
    print(changed)#########################################################
    changed = 0
    inp = input()
print("Overall accuracy of the classifier: {}".format(overallAccuracy/len(testdata.keys())))

for fd in testdata:
    testdata[fd].close()

for fd in files:
    files[fd].close()
