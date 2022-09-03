import streamlit as st
from datasets import load_dataset
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
from sklearn.cluster import KMeans
from bertopic._ctfidf import ClassTFIDF
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

api_key = 'WKsYz7mFBHEDsI7xsZ9KJP3WPMm5txIjKmU0eBTK'
co = cohere.Client(api_key)
title = 'Some title needs to be inputted here'

def get_dataset(df, text, title):
  df.rename(columns={text: 'text', title: 'title'}, inplace=True)
  df = df[['title', 'text']]
  max_length = max(1000, df.shape[0])
  df = df.head(max_length)
  return df

def get_keywords(df, n_clusters=8, chart_title='This is the title'):
  def get_embeddings(df):
    embeds = co.embed(texts=list(df['text']),
                    model='large',
                    truncate='LEFT').embeddings
    reducer = umap.UMAP(n_neighbors=100) 
    umap_embeds = reducer.fit_transform(embeds)
    return (embeds, umap_embeds)

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

def main():
  s = load_dataset('snli')
  s.set_format('pandas')
  df = s['train'][:]
  df = get_dataset(df, text='premise', title='hypothesis')

  st.title('Cohere Embed Viewer')

  app_mode = st.sidebar.selectbox('Mode', ['About', 'EDA', 'Cluster Analysis', 'Search'])

  if app_mode == "About":
      st.markdown('')

      st.markdown(
          """
          <style>
          [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
              width: 350px
          }
          [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
              width: 350px
              margin-left: -350px
          }
          </style>
          """,

          unsafe_allow_html=True,
      )
      st.markdown('')
      st.markdown("Visualize your data using Cohere's embed")
      uploaded_file = st.file_uploader("Choose a file")
      if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

  elif app_mode == 'EDA':
    st.sidebar.subheader(' Quick  Explore')
    st.markdown("Tick the box on the side panel to explore the dataset.")
    if st.sidebar.checkbox('Basic Info'):
        if st.sidebar.checkbox("Show Columns"):
            st.subheader('Show Columns List')
            all_columns = df.columns.to_list()
            st.write(all_columns)

        if st.sidebar.checkbox('Overview'):
            st.subheader('File contents')
            st.write(df)
        if st.sidebar.checkbox('Missing Values?'):
            st.subheader('Missing values')
            st.write(df.isnull().sum())

  elif app_mode == 'Cluster Analysis':
    df = get_dataset(df, text='review_body', title='review_title')
    df, chart_title = get_keywords(df, n_clusters=8, chart_title='This is the Amazon dataset')
    selection = alt.selection_multi(fields=['keywords'], bind='legend')
    chart = alt.Chart(df).transform_calculate(
        url='https://news.ycombinator.com/item?id=' + alt.datum.id
    ).mark_circle(size=60, stroke='#666', strokeWidth=1, opacity=0.3).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        href='url:N',
        color=alt.Color('keywords:N', 
                        legend=alt.Legend(columns=1, symbolLimit=0, labelFontSize=14)
                      ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=['title', 'keywords', 'cluster']
    ).properties(
        width=800,
        height=500
    ).add_selection(
        selection
    ).configure_legend(labelLimit= 0).configure_view(
        strokeWidth=0
    ).configure(background="#FAFAFA").properties(
        title=chart_title
    )
    #chart.interactive()
    st.altair_chart(chart, use_container_width=True)
  elif app_mode == 'Search':
    pass
    # Only this bit    

if __name__ == '__main__':
  main()



