from sklearn.feature_extraction.text import TfidfVectorizer
from fileReader import trainData,testData,writeToCsv
import nltk
from nltk import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import json
from urlextract import URLExtract
import tldextract
import emot
import pickle
import string
import numpy as np
from featureExtractor import extractor

POSDICT = {"\'\'" : 0,
           "(" : 1,
           ")" : 1,
           "," : 1,
           "--" :1,
           "." : 1,
           ":" : 1,
           "CC" : 1,
           "CD" : 1,
           "DT" : 1,
           "EX" : 1,
           "FW" : 1,
           "IN" : 1,
           "JJ" : 1,
           "JJR" : 1,
           "JJS" : 1,
           "LS" : 1,
           "MD" : 1,
           "NN" : 1,
           "NNP" : 1,
           "NNPS" : 1,
           "NNS" : 1,
           "PDT" : 1,
           "POS" : 1,
           "PRP" : 1,
           "PRP$" : 1,
           "RB" : 1,
           "RBR" : 1,
           "RBS" : 1,
           "RP" : 1,
           "SYM" : 1,
           "TO" : 1,
           "UH" : 1,
           "VB" : 1,
           "VBD" : 1,
           "VBG" : 1,
           "VBN" : 1,
           "VBP" : 1,
           "VBZ" : 1,
           "WDT" : 1,
           "WP" : 1,
           "WP$" : 1,
           "WRB" : 1,
           "``": 1,
           "$" : 1,
           "#" : 1}

co = 0
for k in POSDICT.keys():
    POSDICT[k] = co
    co += 1

helper = nltk.help.upenn_tagset()

linkHash = pickle.load(open("URLCache_new.json", 'rb'))
print("cache load finish")
ext = URLExtract()

extract = extractor()


def getLink(h):
    if h in linkHash:
        return linkHash.get(h)

    return tldextract.extract(h).domain



def emExtract(texts1, texts2):
    emDict = {}
    count = 0
    for text in texts1:
        em = emot.emoticons(text)
        try:
            list = em.get('value')
            for e in list:
                if e not in emDict:
                    emDict[e] = count
                    count += 1
        except:
            pass

    for text in texts2:
        em = emot.emoticons(text)
        try:
            list = em.get('value')
            for e in list:
                if e not in emDict:
                    emDict[e] = count
                    count += 1
        except:
            pass

    saveEmD("emD.dict", emDict)

def loadEmD(fileName):
    dic = pickle.load(open(fileName, 'rb'))
    return dic

eDic = loadEmD("emD.dict")

def saveEmD(fileName, obj):
    pickle.dump(obj, open(fileName, 'wb'), 2)

def emVec(texts):
    vecs =[]
    for text in texts:
        vec = [0] * eDic.keys().__len__()
        em = emot.emoticons(text)
        try:
            ed = em.get('value')
            for s in ed:
                if s in eDic:
                    vec[eDic.get(s)] += 1
        except:
            pass

        vecs.append(vec)

    return np.array(vecs)

def tokenize(texts):
    new_texts = []
    count = 0
    pv = []

    for text in texts:
        posVec = [0] * POSDICT.keys().__len__()
        #text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

        links = ext.find_urls(text)
        print(links)
        for l in links:
            try:
                text = re.sub(l, getLink(l), text)
            except:
                pass

        tokens = nltk.word_tokenize(text)

        uniqueToken = {}
        for item in tokens:
            if PorterStemmer().stem(item) not in uniqueToken:
                uniqueToken[item] = 1

        VNvalue = uniqueToken.keys().__len__()/tokens.__len__()
        s = nltk.pos_tag(tokens)
        weight = 1
        for element in s:
            try:
                posVec[POSDICT.get(element[1])] += weight
                weight = weight*1.1
            except:
                pass

        posVec = [float(x)/sum(posVec) for x in posVec]
        posVec.append(VNvalue)
		
        #stems = []
        #for item in tokens:
        #    if len(item) >= 1:
        #        stems.append(PorterStemmer().stem(item))
        #space = " "
        #text = space.join(stems)
        new_texts.append(text)
        pv.append(posVec)
        #print('break')
    return new_texts, pv

td = trainData(threshold=0)
label, rawData = td.getLabelsAndrawData()

#X_train,X_test,y_train,y_test = train_test_split(rawData, label, test_size=0.25, random_state=32)

t, pv = tokenize(rawData)
tf = np.array(extract.batchProduceFixFeatureVec(rawData))
vec = TfidfVectorizer(min_df=5, max_df=20000, stop_words='english',norm='l2', ngram_range=(1,2))
X = vec.fit_transform(t).toarray()
X = np.concatenate((X, tf),axis=1)
X = np.concatenate((X, pv), axis=1)
#X = np.concatenate((X,emVec(X_train)),axis=1)
print(X)
print("break")

#clf = RandomForestClassifier(n_estimators=100, n_jobs=4, verbose=2)
clf = LinearSVC(verbose=2)
#clf = MultinomialNB()
#clf = LogisticRegression(verbose=1,n_jobs=4,solver='sag')
clf.fit(X, label)
td = testData().getAllTweets()
t, pv = tokenize(td)
testSet = extract.batchProduceFixFeatureVec(td)
testSet = np.array(testSet)
test_x = vec.transform(t).toarray()
test_x = np.concatenate((test_x, testSet), axis=1)
test_x = np.concatenate((test_x, pv), axis=1)
#test_x = np.concatenate((test_x, emVec(X_test)), axis=1)
pre = clf.predict(test_x)

writeToCsv(pre)
#print(confusion_matrix(y_test, pre))
#print(accuracy_score(y_test, pre))