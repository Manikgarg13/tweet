
# coding: utf-8

# # Imports

# In[56]:

import re


# In[57]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[58]:

import nltk
from nltk.corpus import stopwords


# In[59]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# In[1]:

TRAIN_PORTION = 0.9


# # Loading Dataset

# In[61]:

dataset_path=r"E:\\training.1600000.processed.noemoticon.csv"
df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])


# In[62]:

df.head()


# # preprocessing for label
# 
# ## -1 as negative, 0 as natural and 1 as positive

# In[63]:

decode_map = {0: -1, 2: 0, 4: 1}
df.target = df.target.apply(lambda x: decode_map[x])


# In[64]:

df.head()


# In[66]:

df.target.value_counts()


# # Clean Data

# In[67]:

def filter_stopwords(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text


# In[68]:

df.text = df.text.apply(filter_stopwords)


# In[69]:

df.head()


# In[70]:

vectorizer = TfidfVectorizer()
word_frequency = vectorizer.fit_transform(df.text)


# In[71]:

len(vectorizer.get_feature_names())


# # Split train and test data

# In[72]:

sample_index = np.random.random(df.shape[0])
X_train, X_test = word_frequency[sample_index <= TRAIN_PORTION, :], word_frequency[sample_index > TRAIN_PORTION, :]
Y_train, Y_test = df.target[sample_index <= TRAIN_PORTION], df.target[sample_index > TRAIN_PORTION]
print(X_train.shape,Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[73]:

clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(X_train, Y_train)


# In[74]:

Y_predit = clf.predict(X_test)


# # Accuracy

# In[75]:

accuracy_score(y_true=Y_test, y_pred=Y_predit)


# In[ ]:




# In[ ]:



