
# Momenton Data Challenge

### Problem Statement: 
You have been engaged to provide insights for a movie production company. They want to understand the most popular movie genres, year by year, for the past decade by using user rating from tweets.

## Movie Tweetings Data Exploration

### Libraries


```python
##Loading all the required libraries
%matplotlib inline
import pandas as pd
import numpy as np
#import random
#import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
```


```javascript
%%javascript
IPython.OutputArea.auto_scroll_threshold = 9999
```


    <IPython.core.display.Javascript object>


### Data

Movie Tweetings dataset provided: https://github.com/momenton/MovieTweetings/tree/master/snapshots/100K


```python
# Reading the rating data file and storing it into dataframes
rating_file = 'F:/Projects/Movie_Reccomender_Systems/momenton-code-test-movietweetings-master/snapshots/100K/ratings.dat'

# data format in rating file: UserID::MovieID::Rating::Timestamp
df_rating = pd.read_csv(rating_file, sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
df_rating.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1074638</td>
      <td>7</td>
      <td>1365029107</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1853728</td>
      <td>8</td>
      <td>1366576639</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reading the movie data file and storing it into dataframes
movies_file = 'F:/Projects/Movie_Reccomender_Systems/momenton-code-test-movietweetings-master/snapshots/100K/movies.dat'

# data format in movie file: MovieID::Title(movie_year)::Genres
df_movie = pd.read_csv(movies_file, sep='::', header=None, names=['movie_id', 'title', 'genre'])
df_movie.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2844</td>
      <td>FantÃ´mas - Ã€ l'ombre de la guillotine (1913)</td>
      <td>Crime|Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4936</td>
      <td>The Bank (1915)</td>
      <td>Comedy|Short</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reading the user data file and storing it into dataframes
users_file = 'F:/Projects/Movie_Reccomender_Systems/momenton-code-test-movietweetings-master/snapshots/100K/users.dat'

# data format in user file: UserID::MovieID
df_users = pd.read_csv(users_file, sep='::', header=None, names=['user_id','movie_id'])
df_users.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18405182</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31260677</td>
    </tr>
  </tbody>
</table>
</div>



### Rating Data Exploration

First we'll explore the rating data and laverage findings to get the insights from it. Let's have a look at few entries from rating data.


```python
df_rating.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1074638</td>
      <td>7</td>
      <td>1365029107</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1853728</td>
      <td>8</td>
      <td>1366576639</td>
    </tr>
  </tbody>
</table>
</div>



Let's have a look at the statistical summary in rating data


```python
df_rating.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 4 columns):
    user_id      100000 non-null int64
    movie_id     100000 non-null int64
    rating       100000 non-null int64
    timestamp    100000 non-null int64
    dtypes: int64(4)
    memory usage: 3.1 MB
    


```python
# Checking the missing values in the rating data
df_rating.isna().sum()
```




    user_id      0
    movie_id     0
    rating       0
    timestamp    0
    dtype: int64



There are no missing values in the data and from statistical summary we can see, there is 1M records of user ratings with 4 key attributes: user id, movie id, rating and timestamp. The timestamp column is integer so We will convert it to datetinme type and extract year and month for further analysis and sort the records by timestamp.


```python
# trasform timestamp atrribute from integer to datetime and exract year and month
df_rating['timestamp'] = df_rating['timestamp'].apply(datetime.fromtimestamp)
df_rating['year'] = df_rating['timestamp'].dt.year
df_rating['month'] = df_rating['timestamp'].dt.month
df_rating['date'] = df_rating['timestamp'].dt.date
df_rating = df_rating.sort_values('timestamp').reset_index(drop=True)
```


```python
df_rating.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>year</th>
      <th>month</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12620</td>
      <td>2171847</td>
      <td>6</td>
      <td>2013-03-01 01:38:27</td>
      <td>2013</td>
      <td>3</td>
      <td>2013-03-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7527</td>
      <td>444778</td>
      <td>8</td>
      <td>2013-03-01 01:43:44</td>
      <td>2013</td>
      <td>3</td>
      <td>2013-03-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Years:', df_rating.year.unique())
print('Months', df_rating.month.unique())
print('#Unique users:', df_rating.user_id.nunique())
print('#Unique movies:', df_rating.movie_id.nunique())
print('Unique ratings:', df_rating.rating.unique())
print('Average rating:', round(df_rating.rating.mean(), 2))
print('Frequency of each rating value:')
print(df_rating['rating'].value_counts())
```

    Years: [2013]
    Months [3 4 5 6 7 8 9]
    #Unique users: 16554
    #Unique movies: 10506
    Unique ratings: [ 6  8  7  5  4  9  2 10  3  1  0]
    Average rating: 7.32
    Frequency of each rating value:
    8     24145
    7     22229
    9     14005
    6     12944
    10    12392
    5      6726
    4      3367
    3      1844
    1      1212
    2      1124
    0        12
    Name: rating, dtype: int64
    

We can see that user rating data is for the period of March to September 2013.  
There are 16554 unique users who have rated 3706 movies in total.  
The rating values are ranging from 0 to 10 with average rating around 7.32 and most frequent rating value is 8.

Also, There is only one year of user ratings data available so let's explore the user rating per month. 


```python
df_temp = df_rating[['month', 'rating']].groupby(['month']).count().reset_index()
df_temp['rating'] = df_temp['rating'] / 10
df_temp.plot.bar(x='month', y='rating', title='Average number of ratings per month', figsize=(10, 5));
```


![png](output_19_0.png)


For the year 2013, the average user rating for the month of Sepetmeber is very less as compared to the rest of the months in the rating data. This shows that only 7.4% users have rated movies for September month.


```python
df_rating['rating'].hist(figsize=(10, 5), linewidth = 1.5, bins=25);
plt.xlim(xmin=0)
plt.show()
```


![png](output_21_0.png)


From this histogram plot we can see that the most common rating value is 8 followed by 7. ratings 6, 9 and 10 are approximately equal about 45%. We can also conclude that most of people giverating 10. Since, the rating data is only available for about 7 months in 2013 year. So, this data is not accurate 100%.


```python
df_rating['rating'].hist(by=df_rating['month'], figsize=(15, 10), bins=25);
```


![png](output_23_0.png)


We can see similar distribution for all the months given the data


```python
df_rating['user_id'].value_counts().nlargest(n=10)
```




    2850     320
    16036    308
    4396     285
    8822     274
    15289    240
    10728    212
    4776     201
    15651    198
    7180     190
    13067    185
    Name: user_id, dtype: int64



We can see that user with id 2850 is the top users with 320 ratings. We can see the distribution of number of reviews per user using box plot and histogram.


```python
df_rating['user_id'].value_counts().plot.box(figsize=(10, 5));
```


![png](output_27_0.png)


We can see that median is around 10. We can users with number of reviews more than extreme value(approximately 30).


```python
df_rating['user_id'].value_counts().hist(figsize=(10, 5));
```


![png](output_29_0.png)


From the histogram we can observe that most of the people(around 16000) are having number of reviews in the range of 1 to 30. Around 500 people are having 30-60 reviews. 50(approximate) people are having more than 70 reviews.

## Movie Data Exploration

Let's have a look at few records in the data


```python
df_movie.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2844</td>
      <td>FantÃ´mas - Ã€ l'ombre de la guillotine (1913)</td>
      <td>Crime|Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4936</td>
      <td>The Bank (1915)</td>
      <td>Comedy|Short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4972</td>
      <td>The Birth of a Nation (1915)</td>
      <td>Drama|History|Romance|War</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5078</td>
      <td>The Cheat (1915)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6684</td>
      <td>The Fireman (1916)</td>
      <td>Short|Comedy</td>
    </tr>
  </tbody>
</table>
</div>



We can see movie release year is appended to tile of the movie, so we'll split the year and store it in release_year dataframe


```python
df_movie['release_year'] = df_movie.title.str.extract("\((\d{4})\)", expand=True).astype(str)
df_movie['release_year'] = pd.to_datetime(df_movie.release_year, format='%Y')
df_movie['release_year'] = df_movie.release_year.dt.year
df_movie['title'] = df_movie.title.str[:-7]
```


```python
# Statistical summary of movie data
df_movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10506 entries, 0 to 10505
    Data columns (total 4 columns):
    movie_id        10506 non-null int64
    title           10506 non-null object
    genre           10440 non-null object
    release_year    10506 non-null int64
    dtypes: int64(2), object(2)
    memory usage: 328.4+ KB
    


```python
# Checking the missing values in the rating data
df_movie.isna().sum()
```




    movie_id         0
    title            0
    genre           66
    release_year     0
    dtype: int64



As we can see there are 66 missing values in genre. Replacing missing values with most common genre for each release_year will skew the data towards most common genre for that release_year which doesn't make any sense. So, We'll drop the na values from genre.


```python
#Droping the missing(NA) values
df_movie = df_movie.dropna(axis=0, how='any')
```


```python
df_movie.isna().sum()
```




    movie_id        0
    title           0
    genre           0
    release_year    0
    dtype: int64




```python
# Printing unique genres
genres_unique = pd.DataFrame(df_movie.genre.str.split('|').tolist()).stack().unique()
print('Unique genres', genres_unique)
```

    Unique genres ['Crime' 'Drama' 'Comedy' 'Short' 'History' 'Romance' 'War' 'Biography'
     'Sport' 'Horror' 'Action' 'Adventure' 'Family' 'Fantasy' 'Mystery'
     'Thriller' 'Documentary' 'Western' 'Sci-Fi' 'Musical' 'Film-Noir'
     'Animation' 'Music' 'Adult' 'News']
    


```python
genres_unique = pd.DataFrame(genres_unique, columns=['genre'])
genres_unique
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Crime</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>4</th>
      <td>History</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Romance</td>
    </tr>
    <tr>
      <th>6</th>
      <td>War</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Biography</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sport</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Horror</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Action</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Family</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Mystery</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Western</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sci-Fi</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Musical</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Film-Noir</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Animation</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Music</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Adult</td>
    </tr>
    <tr>
      <th>24</th>
      <td>News</td>
    </tr>
  </tbody>
</table>
</div>



Creating dummy column for each column with boolean values


```python
df_movie = df_movie.join(df_movie.genre.str.get_dummies().astype(bool))
```


```python
df_movie.drop('genre', inplace=True, axis=1)
```


```python
df_movie.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>release_year</th>
      <th>Action</th>
      <th>Adult</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>...</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>News</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2844</td>
      <td>FantÃ´mas - Ã€ l'ombre de la guillotine</td>
      <td>1913</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4936</td>
      <td>The Bank</td>
      <td>1915</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4972</td>
      <td>The Birth of a Nation</td>
      <td>1915</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5078</td>
      <td>The Cheat</td>
      <td>1915</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6684</td>
      <td>The Fireman</td>
      <td>1916</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
df_movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10440 entries, 0 to 10505
    Data columns (total 28 columns):
    movie_id        10440 non-null int64
    title           10440 non-null object
    release_year    10440 non-null int64
    Action          10440 non-null bool
    Adult           10440 non-null bool
    Adventure       10440 non-null bool
    Animation       10440 non-null bool
    Biography       10440 non-null bool
    Comedy          10440 non-null bool
    Crime           10440 non-null bool
    Documentary     10440 non-null bool
    Drama           10440 non-null bool
    Family          10440 non-null bool
    Fantasy         10440 non-null bool
    Film-Noir       10440 non-null bool
    History         10440 non-null bool
    Horror          10440 non-null bool
    Music           10440 non-null bool
    Musical         10440 non-null bool
    Mystery         10440 non-null bool
    News            10440 non-null bool
    Romance         10440 non-null bool
    Sci-Fi          10440 non-null bool
    Short           10440 non-null bool
    Sport           10440 non-null bool
    Thriller        10440 non-null bool
    War             10440 non-null bool
    Western         10440 non-null bool
    dtypes: bool(25), int64(2), object(1)
    memory usage: 901.1+ KB
    


```python
# Merging the rating and movie data to get desired attributes in single dataframe for better analysis
movie_data = pd.merge(df_rating, df_movie, on='movie_id')
```


```python
movie_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>year</th>
      <th>month</th>
      <th>date</th>
      <th>title</th>
      <th>release_year</th>
      <th>Action</th>
      <th>...</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>News</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12620</td>
      <td>2171847</td>
      <td>6</td>
      <td>2013-03-01 01:38:27</td>
      <td>2013</td>
      <td>3</td>
      <td>2013-03-01</td>
      <td>Dead Mine</td>
      <td>2012</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3892</td>
      <td>2171847</td>
      <td>6</td>
      <td>2013-05-16 23:19:38</td>
      <td>2013</td>
      <td>5</td>
      <td>2013-05-16</td>
      <td>Dead Mine</td>
      <td>2012</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10908</td>
      <td>2171847</td>
      <td>5</td>
      <td>2013-05-25 16:44:19</td>
      <td>2013</td>
      <td>5</td>
      <td>2013-05-25</td>
      <td>Dead Mine</td>
      <td>2012</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>314</td>
      <td>2171847</td>
      <td>3</td>
      <td>2013-05-31 23:46:33</td>
      <td>2013</td>
      <td>5</td>
      <td>2013-05-31</td>
      <td>Dead Mine</td>
      <td>2012</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7250</td>
      <td>2171847</td>
      <td>3</td>
      <td>2013-06-27 19:24:46</td>
      <td>2013</td>
      <td>6</td>
      <td>2013-06-27</td>
      <td>Dead Mine</td>
      <td>2012</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
# Calculate the average ratings for each release year
ratings_mean_count = pd.DataFrame(movie_data.groupby('release_year')['rating'].mean())
```


```python
# Calculate the total number of ratings for each release year
ratings_mean_count['rating_count'] = pd.DataFrame(movie_data.groupby('release_year')['rating'].count())
```


```python
# Store mean count and total number of ratings count in ratings_mean_count for further analysis 
ratings_mean_count.sort_values( by='release_year', ascending=False).reset_index().head(11)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_year</th>
      <th>rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013</td>
      <td>7.159749</td>
      <td>32720</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>7.115258</td>
      <td>22480</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011</td>
      <td>7.092317</td>
      <td>6976</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>7.237633</td>
      <td>4326</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>7.309385</td>
      <td>3090</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2008</td>
      <td>7.430493</td>
      <td>2453</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2007</td>
      <td>7.414024</td>
      <td>2239</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2006</td>
      <td>7.563572</td>
      <td>1982</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005</td>
      <td>7.355049</td>
      <td>1535</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2004</td>
      <td>7.560631</td>
      <td>1839</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2003</td>
      <td>7.644167</td>
      <td>1363</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plotting number of ratings for each year for the past decade (2003-2013)
ratings_mean_count.reset_index().sort_values( by='release_year', ascending=False).head(11).plot.bar(x='release_year', y='rating_count', title='Number of ratings per year', figsize=(20, 3))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fa210637b8>




![png](output_52_1.png)



```python
#Plotting average ratings for each year for the past decade (2003-2013)
ratings_mean_count.reset_index().sort_values( by='release_year', ascending=False).head(11).plot.bar(x='release_year', y='rating', title='Average ratings per year', ylim=(7, 8), figsize=(20, 3));
```


![png](output_53_0.png)



```python
# Calculating total number of mvies count for each genre in movie data
df_temp = pd.DataFrame(columns=['genre', 'num_movies'])
for genre in genres_unique.genre:
    row = [genre, movie_data[movie_data[genre]==True][['movie_id']].count()]
    df_temp.loc[len(df_temp)] = row
df_temp['num_movies'] = df_temp['num_movies'].astype(np.int32)
df_temp = df_temp.sort_values('num_movies', ascending=False).set_index('genre')
plot_fig = df_temp['num_movies'].plot(kind='bar', figsize=(15,7))
plot_fig.set_title('Number of movies for each genre')
plot_fig.set_xlabel('Genre')
```




    Text(0.5, 0, 'Genre')




![png](output_54_1.png)


We can see that Drama and Thriller are having very high number of movies followed by Action, Comedy, Adventure, Sci-Fi, Crime relatively. Romance, Fantasy and horror have average number of movies. We can also have a look at how the trend(based on number of rating) of each genre changes over time.


```python
# Calculating total number of movies count for each genre for the past decade(2003-2013)
for year in range(2003,2014):
    df_year = movie_data[movie_data['release_year']==year]
    df_temp = pd.DataFrame(columns=['genre', 'num_movies'])
    for genre in genres_unique.genre:
        row = [genre, df_year[df_year[genre]==True][['movie_id']].count()]
        df_temp.loc[len(df_temp)] = row
    df_temp['num_movies'] = df_temp['num_movies'].astype(np.int32)
    df_temp = df_temp.sort_values('num_movies', ascending=False).set_index('genre')
    plot_fig = df_temp['num_movies'].plot(kind='bar', figsize=(15,3))
    plot_fig.set_title('Number of movies for each genre for year %s'%year)
    plot_fig.set_xlabel('Genre')
    plt.show()
```


![png](output_56_0.png)



![png](output_56_1.png)



![png](output_56_2.png)



![png](output_56_3.png)



![png](output_56_4.png)



![png](output_56_5.png)



![png](output_56_6.png)



![png](output_56_7.png)



![png](output_56_8.png)



![png](output_56_9.png)



![png](output_56_10.png)


We can observe that Drama is the most common genre for most of the movies followed by Thriller, Comedy, Action, Crime and this the trend is pretty much same until 2012. But number of Action and Sci-Fi movies has dramatically increased in year 2013. Also, count of horror movies has been high in year 2013, which shows users choices and interests are changing more towards Action, Thriller and Sci-Fi movies. 


```python
# Calculating average rating for each genre
df_temp = pd.DataFrame(columns=['genre', 'avg_rating'])
for genre in genres_unique.genre:
    row = [genre, round(movie_data[movie_data[genre]==True][['rating']].mean(), 2).to_string()[6:]]
    df_temp.loc[len(df_temp)] = row
df_temp['avg_rating'] = pd.to_numeric(df_temp['avg_rating'])
df_temp = df_temp.sort_values('avg_rating', ascending=True).set_index('genre')
plot_fig = df_temp['avg_rating'].plot(kind='barh', figsize=(10,7))
plot_fig.set_title('Avg rating for each genre')
plot_fig.set_xlabel('Average Rating') 
plt.xlim(4,9)
plt.show()
```


![png](output_58_0.png)


We can see that genre Fil-Noir is having high average rating followed by short films, Western and so on. Adult movies is having lowest avrage rating among all genres.But we can notice one thing here, Film-Noir has less number movies to be rated by users so it's average rating is more than other movie genre. This trend is same for Short, Western, Biography, Documentry.

The genres having movies rating between 7 to 7.5 have decent movies count so its average rating makes more sense and looks accurate. The middle rage of genres have more count of movies and ratings so this data isn't biased


```python
# Calculating average rating for each genre for the past decade(2003-2013)
for year in range(2013,2003, -1):
    df_year = movie_data[movie_data['release_year']==year]
    df_temp = pd.DataFrame(columns=['genre', 'avg_rating'])
    for genre in genres_unique.genre:
        row = [genre, round(movie_data[movie_data[genre]==True][['rating']].mean(), 2).to_string()[6:]]
        df_temp.loc[len(df_temp)] = row
    df_temp['avg_rating'] = pd.to_numeric(df_temp['avg_rating'])
    df_temp = df_temp.sort_values('avg_rating', ascending=True).set_index('genre')
    plot_fig = df_temp['avg_rating'].plot(kind='barh', figsize=(10,7))
    plot_fig.set_title('Avg rating for each genre for year %s'%year)
    plot_fig.set_xlabel('Average Rating')
    plt.show()
```


![png](output_60_0.png)



![png](output_60_1.png)



![png](output_60_2.png)



![png](output_60_3.png)



![png](output_60_4.png)



![png](output_60_5.png)



![png](output_60_6.png)



![png](output_60_7.png)



![png](output_60_8.png)



![png](output_60_9.png)


Trend is pretty much simillar for all the genres between last 10 years given the data which can concluded from all the graphs above.


```python
#
plt.figure(figsize=(20,10)) 
for genre in genres_unique.genre:
    df_temp = df_movie[df_movie[genre]==True][['release_year', 'movie_id']]
    df_temp = df_temp.groupby(['release_year']).count().reset_index().rename(columns={'release_year':'year', 'movie_id':'#movies'})
    plt.plot(df_temp['year'], df_temp['#movies'], label=genre)
plt.title('Number of movies released each year in different genres')
plt.legend()
plt.show()
```


![png](output_62_0.png)


We can observe from this graph that the distribution of movies with drama genre has the maximum count followed by comedy and Thriller.


```python
# observing distribution of average ratings for each genres using histogram plot 
df_temp = movie_data[['movie_id','rating']].groupby('movie_id').mean()

# Histogram of all ratings
df_temp.hist(bins=25, grid=False, edgecolor='b', normed=True, label ='Overall', figsize=(15,8))

# KDE plot per genre
for genre in genres_unique.genre:
    df_temp = movie_data[movie_data[genre]==True][['movie_id','rating']].groupby('movie_id').mean()
    df_temp.rating.plot(grid=True, alpha=0.9, kind='kde', label=genre)
plt.legend()
plt.xlim(0,10)
plt.xlabel('Rating')
plt.title('Rating Density plot')
plt.show()
```


![png](output_64_0.png)


We can see that all genres possess a left-skewed distribution with a mean around 7.32, except Musical genre because of low rating in the initial years. Also, Mistery genre is having high rating in the intial years so the average is greater than the distribution in the plot. It is also observed with Biography genre as number of movies is less as compared to the user ratings. Drama is consistently most common genre and has more ratings as well so it has noramal distribution over average ratings. 
