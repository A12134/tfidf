import re
from nltk import ngrams
from nltk import TweetTokenizer
import emot
from w2v_processing import w2vAndGramsConverter
from nltk.corpus import stopwords
import operator
import string
import tldextract
import requests
import time
import json
import os.path

class extractor:
    def __init__(self):
        self.vocab = None
        self.firstVocab = dict()
        self.firstCount = 0
        self.secondVocab = dict()
        self.secondCount = 0
        self.thirdVocab = dict()
        self.thirdCount = 0

        self.averageL = 0

        self.ngramVocab = None
        self.firstNgram = dict()
        self.fnc = 0
        self.secondNgram = dict()
        self.snc = 0

        self.gramrdVocab = None

        self.wV = None
        self.tGr = None
        self.rdGr = None

        self.emDict = dict()
        self.emList = []

        self.hashTag = dict()
        self.hashList = []

        self.sess = requests.session()

        self.linkHash = dict()

    def isRetweet(self, line):
        if re.search(r"[R|r][T|t]:?\s@handle:?\s?", line):
            return 1.
        return 0.

    def removeCommonUsedWord_twoG(self, dic):
        ignoreToken = [
            "to",
            "a",
            'and',
            "i",
            "for",
            "of",
            "in",
            "is",
            "on",
            "you",
            "my",
            "rt",
            "-",
            "at",
            "handle",
            "the",
            "im"
        ]

        new_dic = dict(dic)

        for k in dic.keys():
            for t in ignoreToken:
                if t in k:
                    del new_dic[k]
                    break

        dic = dict(new_dic)

        for k in dic.keys():
            for v in k:
                if len(v) <= 2:
                    del new_dic[k]
                    break
        """
        dic = dict(new_dic)

        for k in dic.keys():
            for v in k:
                if re.search(r"http", v):
                    del new_dic[k]
                    break
        """
        return new_dic

    def removeCommonThreeGram(self, dic):
        ignoreToken = [
            "to",
            "a",
            'and',
            "i",
            "for",
            "of",
            "in",
            "is",
            "on",
            "you",
            "my",
            "rt",
            "-",
            "at",
            "handle",
            "the",
            "im"
        ]

        new_dic = dict(dic)

        for k in dic.keys():
            for t in ignoreToken:
                if t in k:
                    del new_dic[k]
                    break

        dic = dict(new_dic)

        for k in dic.keys():
            for v in k:
                if len(v) <= 2:
                    del new_dic[k]
                    break
        """
        dic = dict(new_dic)

        for k in dic.keys():
            for v in k:
                if re.search(r"http", v):
                    del new_dic[k]
                    break
        """
        return new_dic

    def removeCommonUsedWord_list(self, list):
        ignoreToken = {
            "to": 0,
            "a": 0,
            'and': 0,
            "i": 0,
            "for": 0,
            "of": 0,
            "in": 0,
            "is": 0,
            "on": 0,
            "you": 0,
            "my": 0,
            "rt": 0,
            "-": 0,
            "at": 0,
            "handle": 0,
            "ihe": 0,
            "im": 0
        }

        retList = []

        for i in list:
            if ignoreToken.get(i) is None and len(i) > 2:
                retList.append(i)

        return retList



    def removeCommonUsedWord(self, dic):
        ignoreToken = [
            "to",
            "a",
            'and',
            "i",
            "for",
            "of",
            "in",
            "is",
            "on",
            "you",
            "my",
            "rt",
            "-",
            "at",
            "handle",
            "the",
            "im"
        ]

        new_dic = dict(dic)
        for t in ignoreToken:
            if dic.get(t) is not None:
                del new_dic[t]

        dic = dict(new_dic)

        for k in dic.keys():
            if len(k) <= 2:
                del new_dic[k]

        """
        dic = dict(new_dic)

        for k in dic.keys():
            if re.search(r"http", k):
                del new_dic[k]
        """
        return new_dic

    def saveCacheFiles(self):
        print("saving cache data.....")
        with open("URLCache.json", 'w') as fp:
            json.dump(self.linkHash, fp)

    def loadCacheFile(self):
        print("loading cache data.....")
        if os.path.exists("URLCache.json"):
            fp = open("URLCache.json", 'r')
            self.linkHash = json.load(fp)
            print("load finish")
            return
        print("cache data not found")

    def removePuncu(self, line):
        s = line.translate(str.maketrans('','',string.punctuation))
        return s

    def lineNormalization(self, line):
        tknzr = TweetTokenizer()
        norm = w2vAndGramsConverter()

        line = re.sub(r"#\s+", "",line)
        tmp = line.split(" ")
        st = ""
        for t in tmp:
            #reStart = time.time()
            links = re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", t)
            #print("reg time spend: " +  str(time.time() - reStart))
            if links.__len__() > 0:
                #etStart = time.time()
                dom = tldextract.extract(links[0]).domain
                #print("first ext time spend: " + str(time.time() - etStart))
                if dom == 'bit':
                    if self.linkHash.get(links[0]) is None:
                        try:
                            #exreStart = time.time()
                            t = tldextract.extract(self.sess.head(links[0]).headers['location']).domain

                            #print("request and extract time spend:" + str(time.time() - exreStart))
                        except:
                            t = "invaildURL"

                        self.linkHash[links[0]] = t

                    else:
                        t = self.linkHash.get(links[0])
                else:
                    t = dom

            st += t + " "

        line = st


        line = norm.normalizeSentence(line)
        line = self.removePuncu(line)
        line = line.lower()
        sw = stopwords.words('english')
        tokens = tknzr.tokenize(line)

        newT = []
        for t in tokens:
            if t not in sw:
                newT.append(t)


        return newT

    def highFrequencyTokens(self, label, lines):
        currentUsr = label[0]
        gramrd = []
        userrdGram = dict()
        existrdGram = dict()
        gram = []
        userNgram = dict()
        existGram = dict()
        vocab = []
        existvocab = dict()
        userToken = dict()
        for i in range(label.__len__()):
            print("calculating high frequent word features ====>" + str((i+1)*100/label.__len__()) + "%")
            if label[i] is currentUsr:
                if label[i] == '9638':
                    print("break")
                line = lines[i]
                tokens = self.lineNormalization(line)
                #tokens = self.removeCommonUsedWord_list(tokens)
                grams = ngrams(tokens, 2)
                rdgr = ngrams(tokens, 3)

                for rd in rdgr:
                    if userrdGram.get(rd) is None:
                        userrdGram[rd] = 1
                    else:
                        userrdGram[rd] += 1

                for g in grams:
                    if userNgram.get(g) is None:
                        userNgram[g] = 1
                    else:
                        userNgram[g] += 1

                for t in tokens:
                    if userToken.get(t) is None:
                        userToken[t] = 1
                    else:
                        userToken[t] += 1
            else:
                userToken = self.removeCommonUsedWord(userToken)
                userNgram = self.removeCommonUsedWord_twoG(userNgram)
                userrdGram = self.removeCommonThreeGram(userrdGram)
                sortList = sorted(userToken.items(), key=operator.itemgetter(1), reverse=True)
                gramSortList = sorted(userNgram.items(), key=operator.itemgetter(1), reverse=True)
                rdSordList = sorted(userrdGram.items(), key=operator.itemgetter(1), reverse=True)


                if existrdGram.get(rdSordList[0][0]) is None:
                    gramrd.append(rdSordList[0][0])
                    existrdGram[rdSordList[0][0]] = gramrd.__len__()-1

                if existGram.get(gramSortList[0][0]) is None:
                    gram.append(gramSortList[0][0])
                    existGram[gramSortList[0][0]] = gram.__len__()-1
                if existGram.get(gramSortList[1][0]) is None:
                    gram.append(gramSortList[1][0])
                    existGram[gramSortList[1][0]] = gram.__len__()-1
                """
                for x in userToken.keys():
                    if existvocab.get(x) is None:
                        vocab.append(x)
                        existvocab[x] = vocab.__len__() - 1

                """
                if existvocab.get(sortList[0][0]) is None:
                    vocab.append(sortList[0][0])
                    existvocab[sortList[0][0]] = vocab.__len__()-1
                if existvocab.get(sortList[1][0]) is None:
                    vocab.append(sortList[1][0])
                    existvocab[sortList[1][0]] = vocab.__len__()-1
                if existvocab.get(sortList[2][0]) is None:
                    vocab.append(sortList[2][0])
                    existvocab[sortList[2][0]] = vocab.__len__()-1
                if existvocab.get(sortList[3][0]) is None:
                    vocab.append(sortList[3][0])
                    existvocab[sortList[3][0]] = vocab.__len__() - 1
                if existvocab.get(sortList[4][0]) is None:
                    vocab.append(sortList[4][0])
                    existvocab[sortList[4][0]] = vocab.__len__() - 1
                if existvocab.get(sortList[5][0]) is None:
                    vocab.append(sortList[5][0])
                    existvocab[sortList[5][0]] = vocab.__len__() - 1
                if existvocab.get(sortList[6][0]) is None:
                    vocab.append(sortList[6][0])
                    existvocab[sortList[6][0]] = vocab.__len__() - 1
                #if existvocab.get(sortList[7][0]) is None:
                    #vocab.append(sortList[7][0])
                    #existvocab[sortList[7][0]] = vocab.__len__() - 1
                #if existvocab.get(sortList[8][0]) is None:
                    #vocab.append(sortList[8][0])
                    #existvocab[sortList[8][0]] = vocab.__len__() - 1


                userrdGram = dict()
                userNgram = dict()
                userToken = dict()
                currentUsr = label[i]

                line = lines[i]
                tokens = self.lineNormalization(line)
                #tokens = self.removeCommonUsedWord_list(tokens)
                grams = ngrams(tokens, 2)
                rdgr = ngrams(tokens, 3)

                for rd in rdgr:
                    if userrdGram.get(rd) is None:
                        userrdGram[rd] = 1
                    else:
                        userrdGram[rd] += 1

                for g in grams:
                    if userNgram.get(g) is None:
                        userNgram[g] = 1
                    else:
                        userNgram[g] += 1

                for t in tokens:
                    if userToken.get(t) is None:
                        userToken[t] = 1
                    else:
                        userToken[t] += 1

        self.vocab = existvocab
        self.ngramVocab = existGram
        self.gramrdVocab = existrdGram

        self.wV = vocab
        self.rdGr = gramrd
        self.tGr = gram

        print(self.vocab.__len__())
        print(self.ngramVocab.__len__())
        print(self.gramrdVocab.__len__())
        print("break")

    def mentionWordRatio(self, line):
        l = re.findall(r"(?<![R|r][T|t]\s)@handle", line)
        w = self.totalWord(line)
        return l.__len__() / w

    def totalWord(self, line):
        return line.split(" ").__len__()

    def numOfURL(self, line):
        l = re.findall("http", line)
        return l.__len__()

    def hashtagWordRatio(self, line):
        l = re.findall(r"(?<!&)#", line)
        w = self.totalWord(line)
        return l.__len__() / w

    def numOfMoney(self, line):
        l = re.findall(r"\$[1-9][0-9]*", line)
        return l.__len__()

    def lineToVector(self, line):
        return [
            self.isRetweet(line),
            self.mentionWordRatio(line),
            self.totalWord(line),
            self.hashtagWordRatio(line),
            self.numOfMoney(line)
        ]

    def str(self, vec):
        s = ""
        for x in vec:
            s += str(x) + ','
        s = s[:-1]
        return s

    def batchToVector(self, d, usr_flag, save=False, file_name="data"):
        print("convert raw data to vector.....")
        if save:
            print("saving enabled, writing to file: " + file_name)
            save_file = open(file_name, 'w', encoding='utf-8')
            if usr_flag:
                for k in d.keys():
                    for l in d[k]:
                        save_file.write(k + '\t' + self.str(self.lineToVector(l)) + '\n')
            else:
                for l in d:
                    save_file.write(self.str(self.lineToVector(l)) + '\n')

            save_file.close()
            print("saving complete!")

        else:
            if usr_flag:
                new_d = d
                for k in d.keys():
                    for l in d[k]:
                        new_d[k] = self.lineToVector(l)

                return new_d

            else:
                d_arr = []
                for l in d:
                    d_arr.append(self.lineToVector(l))

                return d_arr

    """
    ===============================
            Binary/n Features
    +++++++++++++++++++++++++++++++
    """

    def extractEmoji(self, lines):
        count = 0
        for line in lines:
            line = re.sub(r"http\S+", "", line)
            print("processing emoji: line===>" + str(count))
            li = emot.emoticons(line)
            count += 1
            for v in li.get('value'):
                if self.emDict.get(v) is None:
                    self.emList.append(v)
                    self.emDict[v] = self.emList.__len__()-1

    def emoToVec(self, line):
        line = re.sub(r"http\S+", "", line)
        li = emot.emoticons(line)
        retList = [0]*self.emList.__len__()
        for v in li.get('value'):
            if self.emDict.get(v) is not None:
                retList[self.emDict.get(v)] = 1

        return retList

    def hasEmoji(self, line):
        result = emot.emoticons(line)
        if result.__len__() > 0:
            return 1
        return 0

    # 1: less than 10 tokens(include)
    # 2: less than 20 tokens(include)
    # 3: less than 30 tokens(include)
    # 4: less than 40 tokens(include)
    # 5: more than 40 tokens(exclude)
    def tweetLength(self, line):
        # normalize the line
        w2vLib = w2vAndGramsConverter()
        line = w2vLib.normalizeSentence(line)

        # tokenize sentence
        tnz = TweetTokenizer()
        tokens = tnz.tokenize(line)

        if tokens.__len__() <= 10:
            return 1
        elif tokens.__len__() <= 20:
            return 2
        elif tokens.__len__() <= 30:
            return 3
        elif tokens.__len__() <= 40:
            return 4
        else:
            return 5

    # 0: no URL
    # 1: 1 URL
    # 2: 2 URL
    # 3: 3 URL
    # 4: more than 3 URL
    def getURLFeature(self, line):
        return self.numOfURL(line)

    def newWordUseFeature(self, line):
        w = [0]*self.wV.__len__()
        sG = [0]*self.tGr.__len__()
        trG = [0]*self.rdGr.__len__()

        #line = re.sub(r"#\w+", "", line)

        t = self.lineNormalization(line)
        gr = ngrams(t, 2)
        grrd = ngrams(t, 3)

        for word in t:
            if self.vocab.get(word) is not None:
                w[self.vocab.get(word)] = 1

        for s in gr:
            if self.ngramVocab.get(s) is not None:
                sG[self.ngramVocab.get(s)] = 1

        for tri in grrd:
            if self.gramrdVocab.get(tri) is not None:
                trG[self.gramrdVocab.get(tri)] = 1

        return w+sG+trG

    def wordUseFeature(self, line):
        tokens = self.lineNormalization(line)
        count = 0
        feature = [-1,-1,-1,-1,-1]

        gr = ngrams(tokens, 2)
        grrd = ngrams(tokens, 3)

        for rd in grrd:
            if self.gramrdVocab.get(rd) is not None:
                feature[4] = self.gramrdVocab[rd]
                break

        for g in gr:
            if self.firstNgram.get(g) is not None:
                feature[0] = self.firstNgram[g]
                break
            #elif self.secondNgram.get(g) is not None:
                #feature[1] = self.secondNgram[g]


        for t in tokens:

            if self.firstVocab.get(t) is not None:
                feature[2] = self.firstVocab.get(t)
            elif self.secondVocab.get(t) is not None:
                feature[3] = self.secondVocab.get(t)
            elif self.thirdVocab.get(t) is not None:
                feature[1] = self.thirdVocab.get(t)

        return feature

    def isRT(self, line):
        return self.isRetweet(line)

    def containMoney(self, line):
        if self.numOfMoney(line) > 0:
            return 1
        return 0

    def hasCaptialWord(self, line):
        return re.findall(r"\s[A-Z]{2,}", line).__len__()
        """
        if re.search(r"\s[A-Z]{2,}", line):
            return 1
        return 0
        """

    def hasNowPlaying(self, line):
        if re.search(r"Now playing:\s", line):
            return 1
        return 0

    def useOfPuncs(self, line):
        if re.search(r".{2,}", line) and re.search(r"!{2,}", line):
            return 3
        elif re.search(r".{2,}", line):
            return 2
        elif re.search(r"!{2,}", line):
            return 1
        return 0

    def hasNum(self, line):

        if re.search(r"\s[0-9]*\s", line):
            return 1
        return 0

    def hasMention(self, line):
        return re.findall(r"(?<![R|r][T|t]\s)@handle", line).__len__()
        """
        if re.search(r"(?<![R|r][T|t]\s)@handle", line):
            return 1
        return 0
        """

    def hasRepeatLetters(self, line):
        if re.search(r"\s[a-zA-Z]*([a-zA-Z])\1{1}[a-zA-Z]*\s", line):
            return 1
        return 0

    def extractHashTags(self, lines):
        count = 0
        for line in lines:
            print(count)
            li = re.findall(r"#\w+", line)
            for h in li:
                if self.hashTag.get(h) is None:
                    self.hashList.append(h)
                    self.hashTag[h] = self.hashList.__len__()-1
                    count += 1

        print("break")

    def hashToValue(self, line):
        retVal = [0]*self.hashList.__len__()
        li = re.findall(r"#\w+", line)
        for h in li:
            if self.hashTag.get(h) is not None:
                retVal[self.hashTag.get(h)] = 1

        return retVal

    def hasHash(self, line):
        return re.findall(r"#\w+", line).__len__()
        """
        if re.search(r"#\w+", line):
            return 1
        return 0
        """

    def lineToFixFeatureVec(self, line):
        self.firstVocab = None
        self.secondVocab = None
        self.thirdVocab = None
        self.firstNgram = None
        self.secondNgram = None

        l = [
            self.tweetLength(line),
            self.isRT(line),
            self.hasMention(line),
            self.hasRepeatLetters(line),
            self.hasEmoji(line),
            self.hasNum(line),
            self.containMoney(line),
            self.useOfPuncs(line),
            self.hasHash(line),
            self.hasCaptialWord(line),
            self.COWvalue(line)
        ]

        norm = [float(i)/sum(l) for i in l]

        return norm

    def unloadExt(self):
        self.vocab = None
        self.firstVocab = dict()
        self.firstCount = 0
        self.secondVocab = dict()
        self.secondCount = 0
        self.thirdVocab = dict()
        self.thirdCount = 0

        self.ngramVocab = None
        self.firstNgram = dict()
        self.fnc = 0
        self.secondNgram = dict()
        self.snc = 0

        self.gramrdVocab = None

        self.wV = None
        self.tGr = None
        self.rdGr = None

        self.emDict = dict()
        self.emList = []

        self.hashTag = dict()
        self.hashList = []

        self.sess.close()

        self.linkHash = dict()

    def batchProduceFixFeatureVec(self, lines):
        print("transfering rawdata into vectors.....")
        retArr = []
        for i in range(lines.__len__()):
            print("transfering rawdata into vectors ======>" + str((i+1)*100/lines.__len__()))
            retArr.append(self.lineToFixFeatureVec(lines[i]))

        print("transfer complete!, unloading data")



        return retArr

    # 0: no capital of word
    # 1: 50% of words are start with Capital
    def COWvalue(self, line):
        l = re.findall(r'[A-Z][a-z]+', line)
        w = re.findall(r'[a-zA-Z]+\s', line)
        if w.__len__() >0:
            v = l.__len__() / w.__len__()
            return v
        return 0