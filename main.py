import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

df = pd.read_csv("C:\\Users\\goyal\\Desktop\\project python\\sms_spam\\spam.csv", encoding='latin-1')

#data cleaning

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

df.rename(columns={'v1':'targat','v2':'text'},inplace =True)

encoder = LabelEncoder()
df['targat'] =  encoder.fit_transform(df['targat']) # converting spam and ham in 0 and 1 

# null and duplicate value checking
#a  =df.isnull().sum()
#b = df.duplicated().sum()

df = df.drop_duplicates(keep = 'first')

# counting count of ham and spam
#c = df['tragat'].value_counts()

nltk.download('punkt')
df['num_characters']= df['text'].apply(len)

df['num_words']=  df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df['num_sentances']=  df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

#df[['num_chracters','num_words','num_sentences']].describe()
#df[df['tragat'] == 0 ] [['num_characters','num_words','num_sentences']].describe()
#df[df['tragat'] == 1 ] [['num_characters','num_words','num_sentences']].describe()

#preprocessing

ps = PorterStemmer()

def transform_text(text):
    # lower case
    text = text.lower()
    # converting words into list
    text = nltk.word_tokenize(text)
    y = []
    # removing the special characters
    for i in text: 
        if i.isalnum():
            y.append(i)
    text =y[:]
    y.clear()
    #stopwords removing
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    # stemming the words
    for i in text:
        y.append(ps.stem(i))        
    return " ".join(y)

df['transformed_text']=df['text'].apply(transform_text)
wc = WordCloud(width = 500,height = 500,min_font_size = 10,background_color= 'white')
spam_wc = wc.generate(df[df['targat']==1]['transformed_text'].str.cat(sep=""))

spam_corpus = []
for msg in df[df['targat'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

pd.DataFrame(Counter(spam_corpus).most_common(30))

ham_corpus = []
for msg in df[df['targat']==0][ 'transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

pd.DataFrame(Counter(ham_corpus).most_common(30))


# model building naive bayes

# vectors

tfidf_vectorizer  = CountVectorizer()

X = tfidf_vectorizer.fit_transform(df['transformed_text']).toarray() # type: ignore


y = df['targat'].values

X_train,X_test,y_train,y_test =  train_test_split(X,y , test_size=0.2,random_state=2)

#gnb.fit(X_train,y_train)
#y_pred1 = gnb.predict(X_test)
#print(accuracy_score(y_test,y_pred1))
#print(confusion_matrix(y_test,y_pred1))
#print(precision_score(y_test,y_pred1))

#mnb.fit(X_train,y_train)
#y_pred1 = mnb.predict(X_test)
#print(accuracy_score(y_test,y_pred1))
#print(confusion_matrix(y_test,y_pred1))
#print(precision_score(y_test,y_pred1))

bnb.fit(X_train,y_train)
y_pred2 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

import streamlit as st

st.title(" SMS Spam Checker")

input_sms = st.text_area("Enter your SMS")
if st.button('Predict'):
    transformed = transform_text(input_sms)

    vector_input = tfidf_vectorizer.transform([transformed])

    result = bnb.predict(vector_input)[0] 

    if result == 1:
       st.header("Spam")
    else:
       st.header("Not spam")    

 