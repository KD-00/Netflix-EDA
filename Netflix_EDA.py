
import pandas as pd
import numpy as np

netflix_df = pd.read_csv('netflix_titles.csv')
netflix_df = netflix_df.drop_duplicates('show_id')

show_types = netflix_df.type.unique()
type_of_shows = len(show_types)
print("\nThere are {} types of shows available on Netflix \n 1. {} \n 2. {}".format(type_of_shows,show_types[0],show_types[1]))

no_of_shows = netflix_df.shape[0] 
movies_df = netflix_df[netflix_df.type == 'Movie'].drop_duplicates('title')
no_of_movies = movies_df.shape[0]
TVshows_df = netflix_df[netflix_df.type == 'TV Show'].drop_duplicates('title')
no_of_TVshows = TVshows_df.shape[0]
print('\nThere are a total of {} shows available on Netflix viz. {} Movies {} TV Shows'.format(no_of_shows,no_of_movies,no_of_TVshows))

imdb_movies_df_inp = pd.read_csv('IMDb_movies.csv',low_memory=False)
imdb_movies_df = imdb_movies_df_inp[['imdb_title_id','original_title']]
imdb_movies_df = imdb_movies_df.drop_duplicates('original_title')

imdb_rating_df_inp = pd.read_csv('IMDb_ratings.csv')
imdb_rating_df_inp = imdb_rating_df_inp.drop_duplicates('imdb_title_id')
imdb_rating_df = imdb_rating_df_inp[['imdb_title_id','weighted_average_vote','total_votes']]

imdb_df = imdb_movies_df.merge(imdb_rating_df, on=['imdb_title_id']).drop_duplicates('imdb_title_id')
imdb_df = imdb_df[['original_title','weighted_average_vote','total_votes']]

imdb_df = imdb_df.rename(columns={'original_title':'title','weighted_average_vote':'imdb_rating'})
imdb_df = imdb_df.drop_duplicates('title')

non_rated_movies = np.array(list(set(movies_df.title).difference(set(imdb_df.title))))
print("\nNo of movies without any available rating is",len(non_rated_movies))

netflix_imdb_movies_df = movies_df.merge(imdb_df[['title','imdb_rating','total_votes']], on=['title'], how='left')
netflix_imdb_movies_df = netflix_imdb_movies_df.drop_duplicates('title')

rated_movies = netflix_imdb_movies_df.shape[0] - netflix_imdb_movies_df.imdb_rating.isna().sum()

print("\nTotal Movies: {}".format(no_of_movies))
print("Movies with rating available: {}".format(rated_movies))
print("Movies with no rating available: {}".format(len(non_rated_movies)))

#EDA

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# 1. Availability of Movies of various years.
plt.figure(figsize=(12, 6))
plt.title('Number of Shows V/S Release Year')
plt.xlabel('Release year')
plt.ylabel('Number of shows')
plt.hist(netflix_df.release_year, bins=np.arange(1920,2020,5), color='red')
plt.show()

plt.figure(figsize=(12, 6))
plt.title('Number of Shows V/S Release Year')
plt.xlabel('Release year')
plt.ylabel('Number of shows')
plt.hist([netflix_imdb_movies_df.release_year, TVshows_df.release_year], bins=np.arange(1920,2020,5),stacked=True,
        color=['blue','red'])
plt.legend(['Movies','TV Shows'])
plt.show()


# 2. The boom of Netflix and is it still expanding.
ni_movies_df = netflix_imdb_movies_df.copy()
addition_date_na = ni_movies_df.date_added.isna().sum()
print("\nThere are/is {} movie with date of addition not available".format(addition_date_na))

#dropping the movie with no rating available
drop_index = ni_movies_df[ni_movies_df['date_added'].isnull()].index.tolist()
ni_movies_df2 = ni_movies_df.drop(index=drop_index)

#computing releases vs added date
ni_mov = pd.Series(ni_movies_df2.date_added).value_counts()
ni_mov_df = pd.DataFrame({'date_of_addition':ni_mov.index, 'movies_added':ni_mov.values})
ni_mov_df.date_of_addition = pd.to_datetime(ni_mov_df.date_of_addition)

#Creating new columns for month and year
ni_mov_df['year'] = pd.to_datetime(ni_mov_df.date_of_addition).dt.strftime('%Y')
ni_mov_df['month'] = pd.to_datetime(ni_mov_df.date_of_addition).dt.strftime('%b')

#dropping column date_of_addition
ni_mov_df.drop(columns=['date_of_addition'], inplace=True)
total_movies = ni_mov_df.movies_added.sum()
print("\nTotal no of movies: ",total_movies)

ni_mov_df['year'] = pd.to_numeric(ni_mov_df.year)
ni_mov_df = ni_mov_df.sort_values('year',ascending=False)
heatmap_data = pd.pivot_table(ni_mov_df,values='movies_added', index='month', columns='year', fill_value=0, aggfunc=np.sum)
plt.title("No of Movies Added")
sns.heatmap(heatmap_data, fmt="d", annot=True, cmap="Blues")
plt.show()


# 3. Proportion of various PG rating shows to find the favourable audience as per age.
pg_rating = netflix_df.rating.value_counts()
print(pg_rating)

plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
plt.ylabel('No of shows')
plt.title('PG Ratings')
sns.barplot(x=pg_rating.index, y=pg_rating.values)
plt.show()

# 4. IMDb Ratings of movies
genre_arr = netflix_imdb_movies_df.imdb_rating.value_counts().sort_index()
print(genre_arr)

plt.figure(figsize=(12, 6))
plt.title('Distribution of Ratings of Movies')
plt.xlabel('IMDb Ratings')
plt.ylabel('No of movies')
genre_arr.plot(kind='area', color='purple')
plt.show()

mean_rating = netflix_imdb_movies_df.imdb_rating.mean()
sd_rating = netflix_imdb_movies_df.imdb_rating.std()
med_rating = netflix_imdb_movies_df.imdb_rating.median()
print('\nThe average rating of movies is %.3f'%mean_rating)
print('The standard deviation of ratings is %.3f'%sd_rating)
print("The median of ratings is ",med_rating)

# Answering Questions

# 1. Are older movies having higher rating?
plt.figure(figsize=(12, 6))
plt.title('Distribution of Ratings')
sns.scatterplot(x=netflix_imdb_movies_df.release_year, y=netflix_imdb_movies_df.imdb_rating,color='brown');
plt.show()

# 2. Of the movies with rating of 8 or above, how many are released before 2000?
movie8_df = netflix_imdb_movies_df[netflix_imdb_movies_df['imdb_rating'] >= 8]
movie8_old_df = movie8_df[movie8_df['release_year'] <= 2000].reset_index()
old8_no = movie8_old_df.shape[0]
print("\nThe number of movies released before 2000 and with imdb rating above 8 is", old8_no)


# 3. Is the number of recently released movies increasing on Netflix?
mov2010_df = netflix_imdb_movies_df[netflix_imdb_movies_df['release_year'] >= 2010]
mov_arr = mov2010_df.release_year.value_counts().sort_index()
print(mov_arr)
plt.plot(mov_arr.index, mov_arr.values, 'x--r')
plt.xlabel('Release Year')
plt.ylabel('No of movies')
plt.title("Movie Releases after 2010")
plt.show()

# 4. TV Shows of various countries?
print("\nFor {} TV shows, a country is not listed".format(TVshows_df.country.isna().sum()))
TV_arr = TVshows_df.country.value_counts()
TV_arr = TV_arr[TV_arr.values>20]
print(TV_arr)
plt.figure(figsize=(12, 8))
sns.barplot(x=TV_arr.values, y=TV_arr.index)
plt.title("TV Shows by Country")
plt.xlabel('count')
plt.show()

# 5. Which is the newest movie available on Netflix?
new_mov_df = netflix_imdb_movies_df.sort_values('release_year',ascending=False).head(10)
new_mov = new_mov_df.release_year.value_counts()
new_mov_year = new_mov.index.min()
new_movie = new_mov_df[new_mov_df.release_year == new_mov_year].reset_index()
new_movie.drop(columns=['index','show_id','date_added','type'], inplace=True)
print("\nThe newest movie available is released in the year {}. \nThe newest movie is/are : ".format(new_mov_year))
print(new_movie)