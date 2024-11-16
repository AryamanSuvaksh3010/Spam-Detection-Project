import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data.rename(columns={"v1": "label", "v2": "text"})
data = data[["label", "text"]]
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])  
print(data.head())
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
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
def transform_text(text):
    text = text.lower()  
    text = text.split()  
    text = [word for word in text if word.isalnum()]  
    text = [word for word in text if word not in stopwords]  
    text = [ps.stem(word) for word in text]  
    return " ".join(text)
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data.rename(columns={"v1": "label", "v2": "text"})
data = data[["label", "text"]]
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])  
data['processed_text'] = data['text'].apply(transform_text)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['processed_text']).toarray()
y = data['label']
import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
pickle.dump(model, open('model.pkl', 'wb'))
from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
import streamlit as st  
sms = st.text_input("Enter the message")
if st.button('Predict'):
    transform_sms = transform_text(sms)  
    vector_in = tfidf.transform([transform_sms])  
    res = model.predict(vector_in)[0] 
    st.header("SPAM!" if res == 1 else "Not Spam")  


