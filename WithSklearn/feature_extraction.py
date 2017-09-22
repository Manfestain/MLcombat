# _*_ coding:utf-8 _*_

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def loadDataSet():
    news = fetch_20newsgroups(subset='all')
    return news.data, news.target

def countVectorizer_without_stopwords():
    data, labels = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=33)

    count_vec = CountVectorizer()
    X_count_train = count_vec.fit_transform(X_train)
    X_count_test = count_vec.transform(X_test)

    mnb_count = MultinomialNB()
    mnb_count.fit(X_count_train, y_train)

    print('Accuracy of NBayes(Without filtering stopwords): ',
          mnb_count.score(X_count_test, y_test))
    y_count_predict = mnb_count.predict(X_count_test)
    print(classification_report(y_test, y_count_predict))

def tfidfVectorizer_without_stopwords():
    data, labels = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=33)

    tfidf_vec = TfidfVectorizer()
    X_tfidf_train = tfidf_vec.fit_transform(X_train)
    X_tfidf_test = tfidf_vec.transform(X_test)

    mnb_tfidf = MultinomialNB()
    mnb_tfidf.fit(X_tfidf_train, y_train)

    print('Accuracy of NBayes(TfidfVectoizer without stopwords): ',
          mnb_tfidf.score(X_tfidf_test, y_test))

    y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
    print(classification_report(y_test, y_tfidf_predict))

def countVectorizerAndtfidfVectorizer_with_stopwords():
    data, labels = loadDataSet()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=33)

    count_vec = CountVectorizer(analyzer='word', stop_words='english')
    X_count_train = count_vec.fit_transform(X_train)
    X_count_test = count_vec.transform(X_test)
    mnb_count = MultinomialNB()
    mnb_count.fit(X_count_train, y_train)
    print('Accuracy of NBayes(Without filtering stopwords): ',
          mnb_count.score(X_count_test, y_test))
    y_count_predict = mnb_count.predict(X_count_test)
    print(classification_report(y_test, y_count_predict))

    tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english')
    X_tfidf_train = tfidf_vec.fit_transform(X_train)
    X_tfidf_test = tfidf_vec.transform(X_test)
    mnb_tfidf = MultinomialNB()
    mnb_tfidf.fit(X_tfidf_train, y_train)
    print('Accuracy of NBayes(TfidfVectoizer without stopwords): ',
          mnb_tfidf.score(X_tfidf_test, y_test))
    y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
    print(classification_report(y_test, y_tfidf_predict))

if __name__ == '__main__':
    # countVectorizer_without_stopwords()
    # tfidfVectorizer_without_stopwords()
    countVectorizerAndtfidfVectorizer_with_stopwords()