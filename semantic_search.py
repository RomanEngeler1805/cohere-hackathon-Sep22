import streamlit as st
import cohere
from datasets import load_dataset
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from helper import get_dataset, get_embeddings, get_keywords

@st.cache(allow_output_mutation=True)
def search(df, query):
  api_key = 'dwhPny8kTpkhDNpu05484MtqjFU2QKXeYx9kH6DA'
  co = cohere.Client(api_key)
  title = 'Some title needs to be inputted here'

  # Get embeddings
  embeds, umap_embeds = get_embeddings(df)
  df['x'] = umap_embeds[:,0]
  df['y'] = umap_embeds[:,1]

  # query and embedding
  temp_dict = {'text': query}
  df_query = pd.DataFrame(temp_dict, index=[0])

  # embed query
  query_embed, query_umap_embed = get_embeddings(df_query)

  # create search index
  embeds = np.array(embeds)

  search_index = AnnoyIndex(embeds.shape[1], 'angular')
  # Add all the vectors to the search index
  for i in range(len(embeds)):
      search_index.add_item(i, embeds[i])

  search_index.build(10) # 10 trees
  search_index.save('test.ann')

  # Retrieve the nearest neighbors
  similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                  include_distances=True)
  # Format the results
  results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                              'distance': similar_item_ids[1]})

  # find neighbours
  neighbour = []

  for k in range(len(df)):
    if k in similar_item_ids[0]:
      neighbour.append(1)
    else:
      neighbour.append(0)

  df_neighbour = pd.DataFrame(neighbour, columns=['neighbour'])

  df = df.join(df_neighbour);

  return df
