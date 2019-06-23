import pandas as pd
import numpy as np
from difflib import SequenceMatcher


dbuser = pd.read_excel("userdata.xlsx")
metadata1 = pd.read_csv("movies_metadata.csv", low_memory=False)
metadata = metadata1.head(1000)

#movies list for selecting after login page
metadata = metadata.iloc[0:1000]
metadata = metadata.sample(100)
print(metadata.head(5))
#print(metadata.dtypes)
#print(metadata.loc[:,['imdb_id','title']])
lis1 = metadata.loc[:,['imdb_id','title']]
movlis = lis1.sort_values(by=['title'])
#print(movlis.sort_values(by=['title']))

# Load keywords and credits
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Remove rows with bad IDs.
# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Parse the stringfield features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Print the new features of the first 3 films
print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)

meta = metadata.head(10000)

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(meta['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
meta = meta.reset_index()
indices = pd.Series(meta.index, index=meta['title'])

title = ''
def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:100]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    return meta[['title', 'overview', 'tagline', 'poster_path', 'vote_count', 'vote_average']].iloc[movie_indices]

def user_rec(title):
    rec = get_recommendations(title, cosine_sim2)
    rec = rec[rec['overview'].isna()==False]
    rec = rec.reset_index(drop=True)
    word = dbuser['movie_title'][1]
    li = word.split('|')
    gh = ''
    for i in range(len(li)):
        print(li[i])
        print(metadata1[metadata1['title']==li[i]]['overview'])
        gh=gh+str(metadata1[metadata1['title']==li[i]]['overview'].values)
    s_score = []
    for i in range(len(rec['title'])):
        s_score.append(SequenceMatcher(None, rec['overview'][i],gh).ratio())
    db_new = pd.concat([rec, pd.Series(s_score, name='similarity')], axis=1)
    db_new = db_new.sort_values(by=['similarity'], ascending=False)
    return db_new.head(50)

def K_user(title):
    abc = user_rec(title)
    df_numeric = abc[['vote_count', 'vote_average']]
    from sklearn import preprocessing
    minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric)
    df_numeric_scaled = pd.DataFrame(minmax_processed, index=abc.index, columns=['vote_count', 'vote_average'])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_numeric_scaled)
    len(kmeans.labels_)
    abc['cluster'] = kmeans.labels_
    maxrec = abc.groupby(['cluster'], as_index=False)['vote_average'].mean().max()
    rec_cluster = int(maxrec['cluster'])
    reclist = abc[abc['cluster'] == rec_cluster]
    return reclist
