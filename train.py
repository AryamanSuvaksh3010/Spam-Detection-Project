import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", 
    "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", 
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", 
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", 
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", 
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", 
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", 
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
])

data = pd.read_csv("spam.csv", encoding='latin-1')

def transform_text(text):
    text = text.lower()
    text = list(text.split(' '))
    text = [word for word in text if word not in stopwords]
    return " ".join(text)


print(data.columns)


data = data[['v1', 'v2']]  

data['processed_text'] = data['v2'].apply(transform_text)

data['label'] = data['v1'].map({'ham': 0, 'spam': 1})

X = data['processed_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)


