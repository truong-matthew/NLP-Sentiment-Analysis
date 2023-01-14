import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
reviews_df = pd.read_csv('amazon_reviews.csv')
#reviews_df.info()
#reviews_df.describe()

#sns.countplot(x = reviews_df['rating'])
#plt.show()

reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
#print(reviews_df)

#reviews_df['length'].plot(bins=100, kind='hist')
#plt.show()

sns.countplot(x = reviews_df['feedback'])
#plt.show()

#----------------------
positive = reviews_df[reviews_df['feedback'] == 1]
negative = reviews_df[reviews_df['feedback'] == 0]
pos_sentences = positive['verified_reviews'].tolist()
neg_sentences = negative['verified_reviews'].tolist()

all_pos = " ".join(pos_sentences)
all_neg = " ".join(neg_sentences)


from wordcloud import WordCloud
plt.figure(figsize =(20,20))
plt.imshow(WordCloud().generate(all_pos))


plt.figure(figsize =(20,20))
plt.imshow(WordCloud().generate(all_neg))

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() \
                                    if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)
#print(reviews_df['verified_reviews'][5])
#print(reviews_df_clean[5])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])
print(vectorizer.get_feature_names_out())
#print(reviews_countvectorizer.toarray())
reviews_countvectorizer.shape
reviews = pd.DataFrame(reviews_countvectorizer.toarray())
X = reviews
y = reviews_df['feedback']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))

from sklearn.linear_model import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))