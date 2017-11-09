# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    pass
    token = []
    for genre in movies['genres']:
        token.append(tokenize_string(genre))
    movies['tokens'] = token
    return movies

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    pass
    matrix = []
    all_genre = Counter()
    vocab = {}
    for token in movies['tokens']:
        all_genre.update(set(token))
    all_genres=sorted(all_genre)

    for i in range(len(all_genres)):
        vocab[all_genres[i]] = i


    len_movies = len(movies)
    len_vocab = len(vocab)
    len_all_genres = len(all_genres)

    for token in movies['tokens']:
        token_count = Counter(token)
        #print(token_count.items())
        max_k = token_count[max(token_count.keys())]
        #print(max_k)
        tfidf =[]
        token_count_len = len(token_count)
        #print(token_count_len)
        for all_token in token_count:
            #print(freq)
            tfidf.append(max_k/ max_k * math.log10(len_movies/all_genre[all_token]))
            #print(tfidf)
        row = [0] * token_count_len
        col = []
        for token in token_count:
            #print(token)
            col.append(vocab[token])
            #print(col)
        matrix.append(csr_matrix((tfidf, (row, col)), shape=(1, len_all_genres)))
    movies['features'] = matrix
    #print(vocab)
    #print(movies)
    return movies, vocab



def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    pass
    A = a.toarray()[0]
    B = b.toarray()[0]
    return np.sum(A * B  / (np.sqrt(np.sum(A ** 2)) * np.sqrt(np.sum(B  ** 2))))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    pass
    pred = []
    train = []
    movies_id = []
    ratings_train.index = range(ratings_train.shape[0])
    ratings_test.index = range(ratings_test.shape[0])


    for i in range(len(ratings_train.index)):
        t_d = {}
        t_d['userId'] = ratings_train.loc[i, 'userId']
        t_d['movieId'] = ratings_train.loc[i, 'movieId']
        t_d['rating'] = ratings_train.loc[i, 'rating']
        train.append(t_d)


    for i in range(len(movies)):
        t_d = {}
        t_d['movieId'] = movies.loc[i, 'movieId']
        t_d['features'] = movies.loc[i, 'features']
        movies_id.append(t_d)


    for i in range(len(ratings_test.index)):
        movie_user = ratings_test.loc[i, 'userId']
        movie = ratings_test.loc[i, 'movieId']
        rating = []
        w_avg = []
        right_pos = []
        a = np.matrix([])
        for mov in movies_id:
            if mov['movieId'] == movie:
                a = mov['features']

        for j in train:
            if j['userId'] == movie_user:
                rate = j['rating']
                mid = j['movieId']
                b = np.matrix([])
                for m in movies_id:
                    if m['movieId'] == mid:
                        b = m['features']
                weight = cosine_sim(a, b)
                if weight > 0:
                    rating.append(rate)
                    w_avg.append(weight)
                else:
                    right_pos.append(rate)

        if len(rating) > 0:
            for r in range(len(rating)):
                rating[r] = rating[r] * w_avg[r] / sum(w_avg)
            pred.append(sum(rating))
        else:
            pred.append(np.asarray(right_pos).mean())

    return np.asarray(pred)



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies,vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])

if __name__ == '__main__':
    main()
