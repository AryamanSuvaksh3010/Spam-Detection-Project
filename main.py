import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import string
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Define stopwords
stopwords = ['i', 'me', 'my', ...]  # Add full stopword list here.

# Define text preprocessing function
def transform_text(text):
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords]
    return " ".join(text)

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Rename columns if necessary
data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

# Apply transform_text to preprocess the text
data['processed_text'] = data['text'].apply(transform_text)

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data.rename(columns={"v1": "label", "v2": "text"})  # Rename columns
data = data[['label', 'text']]  # Use only required columns

# Convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess text using the `transform_text` function
data['text'] = data['text'].apply(transform_text)

# Vectorize text
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['text']).toarray()
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save vectorizer and model
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
