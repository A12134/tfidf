from sklearn.feature_extraction.text import TfidfVectorizer
from fileReader import trainData,testData,writeToCsv
import nltk
from nltk.stem.porter import PorterStemmer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

def tokenize(texts):
    new_texts = []
    for text in texts:
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
        tokens = nltk.word_tokenize(text)
        stems = []
        for item in tokens:
            if len(item) >= 2:
                stems.append(PorterStemmer().stem(item))
        space = " "
        text = space.join(stems)
        new_texts.append(text)
        print('break')
    return new_texts

td = trainData()
label, rawData = td.getLabelsAndrawData()

#X_train,X_test,y_train,y_test = train_test_split(rawData, label, test_size=0.25, random_state=32)

t = tokenize(rawData)
vec = TfidfVectorizer(min_df=10, max_df=5000, stop_words='english')
X = vec.fit_transform(t).toarray()
print(X)
print("break")

#clf = RandomForestClassifier(n_estimators=100, n_jobs=4, verbose=2)
clf = LinearSVC(verbose=2)
#clf = MultinomialNB()
#clf = LogisticRegression(verbose=1,n_jobs=4,solver='sag')
clf.fit(X, label)
td = testData().getAllTweets()
t = tokenize(td)
test_x = vec.transform(t).toarray()
pre = clf.predict(test_x)

writeToCsv(pre)
print("finish")
#print(confusion_matrix(y_test, pre))
#print(accuracy_score(y_test, pre))