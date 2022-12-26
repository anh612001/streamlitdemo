
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

imdb_dataset = pd.read_csv('IMDB_Dataset.csv', nrows=2000)



# Extract the list of reviews X
X = imdb_dataset['review'].values.tolist()
# Extract the labels y
y = imdb_dataset['sentiment'].values.tolist()

X_new=[]
n=['<', 'br', '/', '>']
for idx in X:
  word_tokens = word_tokenize(idx)
  # Tao comment moi
  new=[w for w in word_tokens if not w.lower() in stopwords]
  new=' '.join(new)
  X_new.append(new)

  # The whole list of stopwords can be found in stopwords.txt
  remove_word=[w for w in word_tokens if w.lower() in stopwords]
  remove_word=[w for w in word_tokens if w.lower() in n]
  remove_word=' '.join(remove_word)


vectorizer = CountVectorizer()
X_new = vectorizer.fit_transform(X_new)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate by the accuracy score
print(clf.score(X_test, y_test))

filename = 'model1.sav'
joblib.dump(clf, open(filename, "wb"))


loaded_model = joblib.load(filename)

new_review = ["'Doctor Strange' is not a perfect film. If you expect a movie to have depth and the point, you should skip, not only this one, but whole Marvel production."]
new_review_vectorized = vectorizer.transform(new_review)

y_pred = loaded_model.predict(new_review_vectorized)
st.write(y_pred)