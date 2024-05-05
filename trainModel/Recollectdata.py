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

uni = nltk.FreqDist()
bi = nltk.ConditionalFreqDist()
tri = nltk.ConditionalFreqDist()

docs =[]
originalList = []
indexList = []

fd = open("TrainingDataV2/oneRate.txt","r+")

numofNocom = 0
for line in fd.readlines():
    if line == "No Comments\n":
        numofNocom += 1
        continue
    originalList.append(line)   # Append original string to list
    line = line.strip().lower()
    docs.append(word_tokenize(line))    # Append lowered string to docs

fd.seek(0)

for sentence in docs:
    for word in sentence:
        uni[word] += 1

for sentence in docs:
    preceding = "<s>"
    for word in sentence:
        bi[preceding][word] += 1
        preceding = word
    bi[preceding]["</s>"] += 1

for sentence in docs:
    cond1 = "<s>"
    cond2 = "<s>"
    for word in sentence:
        context = " ".join([cond1,cond2])
        tri[context][word] += 1
        cond1 = cond2
        cond2 = word

i = 2
totalcount = 0
score = 0
prolist = []
sentInd = 0
MaxScore = 0
a,b = 0,0
for sentence in docs:
    if totalcount == 1000:
        break
    if len(sentence) in [0,1,2]:
        sentInd += 1
        continue
    for ind in range(len(sentence)-2):
        try:
            w1,w2,w3 = sentence[ind],sentence[ind+1],sentence[ind+2]
            context = " ".join([w1,w2])
            result = tri[context].most_common()
            for wordTuple in result:
                if wordTuple[0] == w3:
                    score += wordTuple[1]
                    break                    
        except:
            break

    scoreperlength = score / len(sentence)
    if MaxScore < scoreperlength: MaxScore = scoreperlength

    if (scoreperlength > 17.52672280896552-8 and scoreperlength<17.52672280896552+8) :
        indexList.append(sentInd)
        totalcount += 1
    if score != 0:      
        prolist.append((score,len(sentence),score/len(sentence)))
    sentInd += 1
    score = 0
        
total_score = 0
for i in prolist:
    total_score += i[2]
print(total_score/len(prolist))
print(len(indexList))

print("totalcount: {}, and max score: {}".format(totalcount,MaxScore))
print(a,b)
fd.close()

fd2 = open("TrainingDataV2/fiveRate.txt","w")
fd3 = open("collectedData/5_5.txt","w")
i = 0
curInd = 0
for sent in originalList:
    if curInd in indexList:
        fd2.write(sent)
    else:
        fd3.write(sent)
    curInd += 1
fd2.close()
fd3.close()
