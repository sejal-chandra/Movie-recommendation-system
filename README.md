# Movie-recommendation-system
Movies recommendation list is based on similarity between the movie given by user to other movies and this list is further divided into clusters.
This is hybrid recommendation system and produces recommendation by filtering movies in 3 stages which are as follows:
1. **Content based filtering:**
Filter movies based on similarity between Credits, Genres and Keywords
2. **Content based filtering:**
Filter movies based on similarity between movies from prevoius list and movies in user's history
3. **Collaborative filtering:**
Movies are divided into clusters based on vote count and average votes. Cluster with most average votes is recommended to user.

[PROJECT DEMO](https://drive.google.com/open?id=11dWNb-_jrrfIK-4PYpKLuv4diaB-LkQK)

## Data Description

This dataset contain 26 million ratings from 270,000 users for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.
* [Dataset link](https://drive.google.com/drive/folders/1JnQXDCsGAb75I4PRRMDHUO0WxmXT-usv?usp=sharing)
* [Dataset Source](https://grouplens.org/datasets/movielens/)

## Technologies Used

### Web Technologies
* Html
* Css
* JavaScript
* Flask

### Machine Learning Library:
* pandas
* numpy
* difflib
* AST
* scikit-learn

### Requirements:
* Python 3.6
