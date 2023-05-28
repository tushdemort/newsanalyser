import pandas as pd
from flair.data import Sentence
from flair.nn import Classifier
tagger = Classifier.load('sentiment')

from textblob import TextBlob as tb
import ssl
from newscatcherapi import NewsCatcherApiClient
import csv
import streamlit as st
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import warnings
warnings.filterwarnings('ignore')


API_KEY = 'YL_4kDF0kT2nUrox1vKfd3rvyD9Txwx8I9-DtKICyKc'

newscatcherapi = NewsCatcherApiClient(x_api_key=API_KEY)

topic=st.text_input("Enter topic",value='Apple')
topic_string=str(topic)

news_articles=newscatcherapi.get_search(q=topic_string,page_size=100,lang='en')



art_ls=[]
for news in news_articles['articles']:
    article={}
    article['title']=news['title']
    article['excerpt']=news['excerpt']
    article['summary']=news['summary']
    art_ls.append(article)


df=pd.DataFrame(art_ls)

df_restructure=df.copy()
df_restructure['title_score']=0
df_restructure['excerpt_score']=0
df_restructure['summary_score']=0


for i in range(len(df_restructure)):
    text=df_restructure['title'][i]
    sentence = Sentence(text)
    tagger.predict(sentence)
    sentence=str(sentence)
    if "NEGATIVE" in sentence:
        df_restructure['title_score'][i]=-1
    elif "POSITIVE" in sentence:
        df_restructure['title_score'][i]=1



for i in range(len(df_restructure)):
    text=df_restructure['excerpt'][i]
    sentence = Sentence(text)
    tagger.predict(sentence)
    sentence=str(sentence)
    if "NEGATIVE" in sentence:
        df_restructure['excerpt_score'][i]=-1
    elif "POSITIVE" in sentence:
        df_restructure['excerpt_score'][i]=1



for i in range(len(df_restructure)):
    text=df_restructure['summary'][i]
    sentence = Sentence(text)
    tagger.predict(sentence)
    sentence=str(sentence)
    if "NEGATIVE" in sentence:
        df_restructure['summary_score'][i]=-1
    elif "POSITIVE" in sentence:
        df_restructure['summary_score'][i]=1



    
    
df_restructure['avg_sentiment']=0
for i in range(len(df_restructure)):
    ab=df_restructure['title_score'][i]
    pr=df_restructure['excerpt_score'][i]
    tl=df_restructure['summary_score'][i]
    avg_sentiment=(ab+pr+tl)/3
    df_restructure['avg_sentiment'][i]=avg_sentiment

opinion_score=df_restructure['avg_sentiment'].mean(axis=0)

opinion='Positive' if opinion_score>0 else "Negative"
if opinion_score==0:
    opinion='Neutral'


output='The current sentiment in the news on '+topic_string+' is '+opinion
st.text(output)
for news in news_articles['articles']:
    t=news['title']
    l=news['link']
    q=str("["+t+"]("+l+")")
    st.write(q)