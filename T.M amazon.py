# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:17:34 2022

@author: lalith kumar
"""

# data scraping from amazon 
# (product name) Boat Rockerz 425 Bluetooth Wireless Over Ear Headphones with Mic and Boat Signature Sound, Beast Mode for Gaming, Enx Tech, ASAP Charge, 25H Playtime, Bluetooth V5.2 (Active Black)

#importing Libraries.
import requests
from bs4 import BeautifulSoup as bs # for web scraping

# by using the below url we are going to scrap the reviews.
#https://www.amazon.in/Rockerz-425-Bluetooth-Headphones-Signature/dp/B09QL3NQHX/ref=sr_1_2_sspa?keywords=headphones&qid=1654584389&sprefix=%2Caps%2C236&sr=8-2-spons&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzR041RzJLNjNLUTZKJmVuY3J5cHRlZElkPUEwMjg3NTM5M0NXMU5ZMklFVDI0TyZlbmNyeXB0ZWRBZElkPUEwOTE3NTY0Mk0yV1RZUVBHMUhPNiZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU&th=1
# Creating empty review list

boat_bluetooth_reviews = []
for i in range(1,40):
    earphones= []
    url = "https://www.amazon.in/Rockerz-425-Bluetooth-Headphones-Signature/dp/B09QL3NQHX/ref=sr_1_2_sspa?keywords=headphones&qid=1654584389&sprefix=%2Caps%2C236&sr=8-2-spons&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzR041RzJLNjNLUTZKJmVuY3J5cHRlZElkPUEwMjg3NTM5M0NXMU5ZMklFVDI0TyZlbmNyeXB0ZWRBZElkPUEwOTE3NTY0Mk0yV1RZUVBHMUhPNiZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU&th=1"+str(i)
    header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36.'}
    response = requests.get(url,headers = header)
    
# Creating soup object to iterate over the extracted content.
soup = bs(response.text,"lxml")

# Extract the content under the specific tag.
reviews = soup.find_all("div",{"data-hook":"review-collapsed"})
for i in range(len(reviews)):
   earphones.append(reviews[i].text)

# Adding the reviews of one page to empty list which in future contains all the reviews.
boat_bluetooth_reviews += earphones
    
    
# Writing reviews in a text file.
with open('boat_bluetooth_reviews.txt','w', encoding = 'utf8') as output:
    output.write(str(boat_bluetooth_reviews))
   
#regular expression
import re 
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download("stopwords")
from nltk.corpus import stopwords

# Joining all the reviews into single paragraph. 
boat_rev_string = " ".join(boat_bluetooth_reviews)

# Change to lower case and removing unwanted symbols.
boat_rev_string = re.sub("[^A-Za-z" "]+"," ",boat_rev_string).lower()
boat_rev_string = re.sub("[0-9" "]+"," ",boat_rev_string)

# words that contained in boat Headphones reviews.
boat_reviews_words = boat_rev_string.split(" ")

# Lemmatizing.
wordnet = WordNetLemmatizer()
boat_reviews_words=[wordnet.lemmatize(word) for word in boat_reviews_words]

# Filtering Stop Words.
stop_words = set(stopwords.words("english"))
stop_words.update(['amazon','product','speaker','boat','bluetooth','bwz','version','neckbandi','op'])
boat_reviews_words = [w for w in boat_reviews_words if not w.casefold() in stop_words]

from sklearn.feature_extraction.text import TfidfVectorizer
# TFIDF: bigram
bigrams_list = list(nltk.bigrams(boat_reviews_words))
bigram = [' '.join(tup) for tup in bigrams_list]
vectorizer = TfidfVectorizer(bigram,use_idf=True,ngram_range=(2,2))
X = vectorizer.fit_transform(bigram)
vectorizer.vocabulary_
sum_words = X.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

#pip install wordcloud
 
import matplotlib.pyplot as plt
from wordcloud import wordcloud
words_dict = dict(words_freq)
wordCloud = wordcloud(height=1400, width=1800)
wordCloud.generate_from_frequencies(words_dict)
plt.title('Most Frequently Occurring Bigrams')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# initialize VADER
sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]
for word in boat_reviews_words:
    if (sid.polarity_scores(word)['compound']) >= 0.25:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['compound']) <= -0.25:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word) 

# Positive word cloud.

boat_pos_in_pos = " ".join ([w for w in pos_word_list])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(boat_pos_in_pos)
plt.title("Positive Words In The Review of boat Headphones")
plt.imshow(wordcloud_pos_in_pos, interpolation="bilinear")

# negative word cloud.

boat_neg_in_neg = " ".join ([w for w in neg_word_list])
wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(oneplus_neg_in_neg)
plt.title("Negative Words In The Review of boat Headphones")
plt.imshow(wordcloud_neg_in_neg, interpolation="bilinear")


#IMPORT PANDAS.
import pandas as pd  
#here convert raw data into dataformat. 
df=pd.DataFrame()
df["Text"]=boat_bluetooth_reviews
df.to_csv(r"E:\data science\amazon.csv",index=True)

reviews=pd.read_csv("E:\data science/amazon.csv")

reviews=[Text.strip() for Text in reviews.Text] # remove both the leading and the trailing characters
reviews=[Text for Text in reviews if Text] # removes empty strings, because they are considered in Python as False
reviews[0:10]

#Emotion Mining & Sentiment Analysis.
from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(reviews))
sentences

sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df

# importing affin dataset.
# Emotion Lexicon - Affin#AFFINITY.
affin=pd.read_csv('E:\data science/Afinn.csv',sep=',',encoding='Latin-1')
affin

# checking affin score.

affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores

# import spacy.

import spacy
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

# Calculating sentiment value.
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

# Plotting the sentiment values.
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])

# Plotting the line plot for sentiment values.
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

# Correlation analysis.
sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word count')

#------------------------------------------------------------------------------
ss




