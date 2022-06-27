
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:14:30 2022

@author: lalith kumar
"""

# Importing Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy

%matplotlib inline
# import  the dataset
df=pd.read_csv('E:\data science\ASSIGNMENTS\ASSIGNMENTS\TEXT MINING\Elon_musk.csv',encoding='Latin-1')
df.drop(['Unnamed: 0'],inplace=True,axis=1)
df

#Text Preprocessing.
# remove both the leading and the trailing characters.
df=[Text.strip() for Text in df.Text] 
# removes empty strings, because they are considered in Python as False
df=[Text for Text in df if Text] 
df[0:10]

# Joining the list into one string/text
df_text=' '.join(df)
df_text

# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
df_tokens=tknzr.tokenize(df_text)
print(df_tokens)

# Again Joining the list into one string/text
df_tokens_text=' '.join(df_tokens)
df_tokens_text

# Remove Punctuations 
no_punc_text=df_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text

# remove https or url within text.
#REGULAR EXPRESSION.
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text

from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens)

# Tokens count
len(text_tokens)

# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I','U0001F525','\x94','U0440','\x93']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)
len(no_stop_tokens)
# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])

# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])
len(stemmed_tokens)
      #or
# Lemmatization
nlp=spacy.load('en_core_web_sm')  
doc=nlp(' '.join(lower_words))
print(doc)

lemmas=[token.lemma_ for token in doc] 
print(lemmas[100:200])

clean_tweets=' '.join(lemmas)
clean_tweets

#Feature Extaction
# Using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)
print(cv.vocabulary_)

print(cv.get_feature_names()[100:200])
print(tweetscv.toarray()[100:200])

print(tweetscv.toarray().shape)

# CountVectorizer with N-grams (Bigrams & Trigrams)
cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)
print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())

# TF-IDF Vectorizer(Term Frequency – Inverse Document Frequency.)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)
print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())

# Generate Word Cloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')    
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)

#Named Entity Recognition (NER)
# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')
from spacy import displacy
one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)

for token in doc_block[100:200]:
    print(token,token.pos_)    

# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])

# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results

# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs')

#Emotion Mining - Sentiment Analysis
from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(tweets))
sentences

sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df

# Emotion Lexicon - Affin#AFFINITY
affin=pd.read_csv('E:\data science/Afinn.csv',sep=',',encoding='Latin-1')
affin

affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores

# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score

# manual testing.
calculate_sentiment(text='great')

# Calculating sentiment value for each sentence.
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']

# checking how many words are there in a sentence.
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']

sent_df.sort_values(by='sentiment_value')

# Sentiment score.
sent_df['sentiment_value'].describe()

# negative sentiment score.
sent_df[sent_df['sentiment_value']<=0]

# positive sentiment score.
sent_df[sent_df['sentiment_value']>0]

# Adding index cloumn.
sent_df['index']=range(0,len(sent_df))
sent_df

# Plotting the sentiment value.
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])

# Plotting the line plot for sentiment value.
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

# Correlation analysis.
sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word count')

#-----------------------------------------------------------------------------













































