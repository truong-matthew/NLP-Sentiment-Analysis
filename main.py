import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
import sklearn


# from jupyterthemes import jtplot
# jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
tweets_df = pd.read_csv('twitter.csv')
tweets_df = tweets_df.drop(['id'], axis=1)

""" 
----------------------
# Common Print Statements to get info on dataframe
----------------------
"""
# print(tweets_df.info())
# print(tweets_df.describe())
# print(tweets_df['tweet'])

# sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap = "Blues")

# Histogram // Countplot
#tweets_df.hist(bins=30, figsize = (13, 5), color = 'g')

#sns.countplot(data=tweets_df['label'], label='Count') #WHY ONLY SHOW 1 COLUMN???
#plt.show()


tweets_df['length'] = tweets_df['tweet'].apply(len)
# print(tweets_df.head())
# print(tweets_df.describe())

k = tweets_df[tweets_df['length'] == 11]['tweet']
practice2 = tweets_df[tweets_df['length'] == 84]['tweet'].iloc[0]  # first entry
# print(practice2)

tweets_df['length'].hist(bins=100, figsize=(13, 5), color='g')
tweets_df['length'].plot(bins=100, kind='hist')
## plt.show()  #THIS SHOWS LENGTHS OF TWEETS IN DATA

# note that loc index an aspect of a df while here we want to create a new/filtered one
negative = tweets_df[tweets_df['label'] == 1]
positive = tweets_df[tweets_df['label'] == 0]

''' wordcloud import not working
sentences = tweets_df['tweet'].tolist()
ll = len(sentences)
sentences_as_one_str = " ".join(sentences)

print(sentences_as_one_str)

from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_str))
'''
# print(string.punctuation)
Test = 'Good Morning Dat Bussy :)... slat AI11!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
#print(Test_punc_removed)
Test_punc_removed_join = ''.join(Test_punc_removed)
#print(Test_punc_removed_join)

# NATURAL LANGUAGE TOOL KIT
from nltk.corpus import stopwords

stopwords.words('english')

Test_punc_removed_join = "I enjoy coding, programming, and AI"
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if
                                word.lower() not in stopwords.words('english')]

# print(Test_punc_removed_join_clean)


# PIPELINE TASK 4
mini_c = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'


def clean_line(sentence):
    filter1 = []
    final = []
    for char in sentence:
        if char not in string.punctuation:
            filter1.append(char)

    filter1 = ''.join(filter1).split()
    for word in filter1:
        if word.lower() not in stopwords.words('english'):
            final.append(word)
    print(final)

challenge = [char  for char in mini_c if char not in string.punctuation ]
challenge = ''.join(challenge)
challenge = [word  for word in challenge.split() if word.lower() not in stopwords.words('english')]
#print(challenge)

#ANSWER = mini challenge teach remove stopwords punctuations (working)
#clean_line(mini_c)


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This paper is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

#print(vectorizer.get_feature_names_out()) #prints every unique word
#print(X.toarray()) #array matching words

mini_c2 = ['Hello World', 'Hello Hello World', 'Hello World world world'] #countvectorizations converts everything to lowercase
#X2 = vectorizer.fit_transform(mini_c2)
#print(X2.toarray())


#TASK 9 CREATE A PIPELINE TO REMOVE PUNCT/STOPWORDS/PERF COUNT VECTORIZATION

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() \
                                    if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
'''
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)
print(tweets_df_clean[5])
print(tweets_df['tweet'][5])
'''

from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(analyzer= message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])

#every unique word in dataset
ksi = vectorizer.get_feature_names_out()

#print(tweets_countvectorizer.toarray())

X = pd.DataFrame(tweets_countvectorizer.toarray())
#print(XX)
y = tweets_df['label']

#Module 11
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))

#classification (this results uses a correct dataset and compares accuracy)
#plt.show()

