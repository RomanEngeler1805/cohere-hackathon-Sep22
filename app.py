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
from PIL import Image
import requests
from helper import get_dataset, get_embeddings, get_keywords
from semantic_search import search
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)


api_key = 'WKsYz7mFBHEDsI7xsZ9KJP3WPMm5txIjKmU0eBTK'
co = cohere.Client(api_key)
title = 'Some title needs to be inputted here'



@st.cache(allow_output_mutation=True)
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

  df = pd.DataFrame({'title': [], 'text': []})
  # title
  col1, _, col2 = st.columns([1,1,15])

  image_cohere = Image.open(requests.get('https://avatars.githubusercontent.com/u/54850923?s=280&v=4', stream=True).raw)
  col1.image(image_cohere, width=80)
  #
  col2.title('Analyze')
  #
  app_mode = st.sidebar.selectbox('Task', ['Import', 'EDA', 'Cluster', 'Search'])

  with st.expander("How this works.", expanded=False):
    st.write(
      """     
      Embeddings are hard to vizualize. Analyze makes it a breeze.
      1. Go ahead and upload a csv file that you want to examine.
          The csv file needs to have atleast 2 columns:
          - The first column being shorter text - a title for example.
          - The second column being the longer text - the body of the text for example.
      2. You then have 3 options:
          - EDA: Get an overview of the file and get some general exploratory data analysis.
          - Cluster: Do some cluster analysis, with keywords generated from the body of the text and using the titles.
          - Search: Query the data and retrieve the closest match.

      To make sure this app works quickly, we only capture the first 500 lines of text.     
      """
    )
    st.markdown('')
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, usecols=[0, 1])

  
   
  if app_mode == "Import":
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
      #with st.expander("Help", expanded=False):
      #  st.write(
      #    """     
      #    Embeddings are hard to vizualize. Analyze makes it a breeze.
      #    1. Go ahead and upload a csv file that you want to examine.
      #       The csv file needs to have atleast 2 columns:
      #       - The first column being shorter text - a title for example.
      #       - The second column being the longer text - the body of the text for example.
      #    2. You then have 3 options:
      #       - EDA: Get an overview of the file and get some general exploratory data analysis.
      #       - Cluster: Do some cluster analysis, with keywords generated from the body of the text and using the titles.
      #       - Search: Query the data and retrieve the closest match.
      #
      #    To make sure this app works quickly, we only capture the first 500 lines of text.     
      #    """
      #)
      #st.markdown('')
      #uploaded_file = st.file_uploader("Choose a file")
      #if uploaded_file is not None:
      #  df = pd.read_csv(uploaded_file, usecols=[0, 1])
      #  st.write(uploaded_file.name)
      #  st.write(dataframe)
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
            st.write(f'The number of lines is {df.shape[0]}. We will only process {min(500, df.shape[0])}')
        if st.sidebar.checkbox('Missing Values?'):
            st.subheader('Missing values')
            st.write(df.isnull().sum())

  elif app_mode == 'Cluster':
    df.columns = ['title', 'text']
    embeds, umap_embeds = get_embeddings(df)
    low, med, high = 1, 8, 10

    with st.expander("Help", expanded=False):
      st.write(
          """     
          One of the ways to determine the optimal number of clusters, is to choose the number corresponding to an elbow, if it exists.
          """
      )
      st.markdown("")
      distortions = []
      nembeds = np.array(embeds)
      for k in range(low, high + 1):
        km = KMeans(n_clusters=k)
        km.fit(nembeds)
        distortions.append(sum(np.min(cdist(nembeds, km.cluster_centers_,
                                              'euclidean'), axis=1)) / nembeds.shape[0])  

      fig = plt.figure(figsize=(10, 4))
      plt.plot(range(low, high + 1), distortions, 'bx-')
      plt.xlabel('Number of clusters')
      plt.ylabel('Distortion')
      plt.title('Determine the optimal number of clusters')
      st.pyplot(fig)

    n_clusters = st.slider('Select number of clusters', low, high, med)
    #st.write('Number of clusters:', n_clusters)
    #df.columns = ['title', 'text']
    #st.write(df.head())
    df = get_dataset(df, text='text', title='title')
    try:
      chart_title = uploaded_file.name.split('.')[0]
    except:
      chart_title = 'Title TBD'
    df, chart_title = get_keywords(df, n_clusters=n_clusters, chart_title=chart_title)
    selection = alt.selection_multi(fields=['keywords'], bind='legend')
    chart = alt.Chart(df).transform_calculate(
        url=alt.datum.id
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
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
  elif app_mode == 'Search':
    # call the search Function
    # TODO: pass query from here
    # call the search Function
    query = st.text_input(label='Search query', value='Show me something important')
    #st.write(df.head())
    df.columns = ['title', 'text']
    df = search(df, query)
    # Plot
    chart = alt.Chart(df).transform_calculate(
        url= alt.datum.id
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
        color=alt.Color('neighbour', scale=alt.Scale(domain=[0, 1], range=['blue', 'red'])),
        tooltip=['title']
    ).properties(
        width=800,
        height=500
    ).configure_legend(labelLimit= 0).configure_view(
        strokeWidth=0
    ).configure(background="#FAFAFA")
    #chart.interactive()
    st.altair_chart(chart.interactive(), use_container_width=True)

if __name__ == '__main__':
  main()



