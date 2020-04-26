#!/usr/bin/env python
# coding: utf-8

# In[1]:

from numpy import argmax
import re
import nltk
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import praw
import h5py
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


# In[2]:


model = load_model('./model_data/model_CNN.h5')
model._make_predict_function()


# In[3]:


with open('./model_data/model_CNN_cache.pickle', 'rb') as f:
    cache = pickle.load(f)


# In[4]:


tk = cache['tokenizer']
le = cache['Label_Encoder']


# In[5]:


# praw instance
reddit = praw.Reddit(client_id='FIvso8ETavDmfw', client_secret='uZy6eZ6fs7PJEblJO2lwbxs1f_U', user_agent='project_script')


# In[6]:


stopwords_set = set(stopwords.words('english'))          # stopwords set
wnl = WordNetLemmatizer()                                # Lemmatizer

def lemmatise(sent):
    """
    Performs lemmatization of text
    """
    sent = nltk.tag.pos_tag(sent)
    sent = [(w,get_wordnet_pos(p)) for (w,p) in sent]
    sent = [wnl.lemmatize(w,tag) for (w,tag) in sent]
    return sent

def get_wordnet_pos(tag):
    """
    returns part_of_speech tag to lemmatizer in 'wordnet' format for lemmatization
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def text_cleaning(sent):
    sent = sent.lower()   #converting to lower case
    sent = re.sub("[^a-zA-Z]+|(pdf|html|www|http|jpg|com)"," ",sent) # removing numbers and punctuation
    sent = sent.split()

    sent = [w for w in sent if len(w)>1] # removing words of length 1
    sent = [w for w in sent if w not in stopwords_set] # removing stopwords
    sent = lemmatise(sent)    # lemmatization
    sent = " ".join(sent)
    return sent

def regex_on_url(text):    # custom regex to extract of only meaningful words from url
    return re.sub("(https://)|[a-z.]+(com)|[^a-zA-Z]|(pdf|html|www|http|jpg)",' ',text)

def clean_url(text):
    text = text.lower()       # converting to lowercase
    text = regex_on_url(text) # regex
    text = text.split()

    text = [w for w in text if len(w)>1] # removing words of length 1
    text = [w for w in text if w not in stopwords_set] # removing stopwords
    text = " ".join(text)
    return text


# In[7]:


def extract_text_from_url(url):
    submission = reddit.submission(url = url)

    title = submission.title
    title = text_cleaning(title)

    body = submission.selftext
    body = text_cleaning(body)

    URL = submission.url
    URL = clean_url(URL)

    text = title + " " + body + " " + URL
    text_list = []
    text_list.append(text)
    return text_list


# In[8]:


def predict_flair(url):
    text = extract_text_from_url(url)
    X_test = tk.texts_to_sequences(text)
    X_test_pad = pad_sequences(X_test,maxlen = 300, padding = 'post',truncating = 'post')
    prediction = model.predict(X_test_pad)
    prediction = argmax(prediction,axis = 1)
    prediction = le.inverse_transform(prediction)[0]

    return prediction
