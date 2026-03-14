import  numpy as np
import  pandas as pd
import re
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# %%
news_df = pd.read_csv('train.csv')

# %%
news_df = news_df.dropna() # remove rows with missing values

# %%
news_df['content'] = news_df['author']+" "+news_df['title']

# %%
# stemming
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]'," ",content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemstd_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemstd_content)
    return stemmed_content

# %%
news_df['content'] = news_df['content'].apply(stemming)

# %%
X = news_df['content'].values
y = news_df['label'].values

# %%
vector = TfidfVectorizer(max_features=5000)
vector.fit(X)
X = vector.transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1)

# %%
model = LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)

# %%
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# %%
st.title('Fake News Detector')
input_text = st.text_input("Enter News Article")

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write("This news article is fake")
    else:
        st.write("This news article is real")






