import streamlit as st
import pandas as pd
import cohere
import umap
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.cluster import KMeans
from bertopic._ctfidf import ClassTFIDF
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

api_key = 'WKsYz7mFBHEDsI7xsZ9KJP3WPMm5txIjKmU0eBTK'
co = cohere.Client(api_key)
title = 'Some title needs to be inputted here'

@st.cache(allow_output_mutation=True)
def get_dataset(df, text, title):
  max_length = 500
  df.rename(columns={text: 'text', title: 'title'}, inplace=True)
  df = df[['title', 'text']]
  max_length = min(max_length, df.shape[0])
  df = df.head(max_length)
  return df

@st.cache(allow_output_mutation=True)
def get_embeddings(df):
    embeds = co.embed(texts=list(df['text']),
                    model='large',
                    truncate='LEFT').embeddings
    reducer = umap.UMAP(n_neighbors=100) 
    umap_embeds = reducer.fit_transform(embeds)
    return (embeds, umap_embeds)

@st.cache(allow_output_mutation=True)
def get_keywords(df, n_clusters=8, chart_title='This is the title'):
  embeds, umap_embeds = get_embeddings(df)
  df['x'] = umap_embeds[:,0]
  df['y'] = umap_embeds[:,1]

  kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
  classes = kmeans_model.fit_predict(embeds)
  documents =  df['title']
  documents = pd.DataFrame({"Document": documents,
                            "ID": range(len(documents)),
                            "Topic": None})
  documents['Topic'] = classes
  documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
  count_vectorizer = CountVectorizer(stop_words="english").fit(documents_per_topic.Document)
  count = count_vectorizer.transform(documents_per_topic.Document)
  words = count_vectorizer.get_feature_names()
  ctfidf = ClassTFIDF().fit_transform(count).toarray()
  words_per_class = {label: [words[index] for index in ctfidf[label].argsort()[-10:]] for label in documents_per_topic.Topic}
  df['cluster'] = classes
  df['keywords'] = df['cluster'].map(lambda topic_num: ", ".join(np.array(words_per_class[topic_num])[:]))
  return df, chart_title
