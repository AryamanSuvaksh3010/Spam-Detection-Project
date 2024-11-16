import streamlit as st
import pickle
import sklearn
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
 "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

import string
string.punctuation


def transform_text(text):
    # convert to lower case
    text = text.lower()

    # tokenize the words
    text = list(text.split(' '))

    # removing all the special characters
    y = []
    for i in text:
        yt = ""
        for j in range(len(i)):
            if i[j].isalnum():
                yt = yt + i[j]
        y.append(yt)
    text = y[:]
    y.clear()

    # removing stop words and punctuation
    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)

    # stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

sms = st.text_input("Enter the message")
if st.button('Predict'):
    # 1. preproccess the text
    transform_sms = transform_text(sms)
    # 2. vectorize
    vector_in = tfidf.transform([transform_sms])
    # 3. predict
    res = model.predict(vector_in)[0]
    # 4. Display
    if res == 1:
        st.header("SPAM!")
    else:
        st.header("not spam")