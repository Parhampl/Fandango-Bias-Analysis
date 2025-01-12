#!/usr/bin/env python
# coding: utf-8

# <span style="color:red; background:pink">Submit the final Notebook after completing all tasks</span>
# 
# # Deadline: 22<sup>th</sup> November (1 Azar)

# # Capstone Project
# ## Overview
# 
# If you are planning on going out to see a movie, how well can you trust online reviews and ratings? *Especially* if the same company showing the rating *also* makes money by selling movie tickets. Do they have a bias towards rating movies higher than they should be rated?
# 
# ### Goal:
# 
# **Your goal is to complete the tasks below based off the 538 article and see if you reach a similar conclusion. You will need to use your pandas and visualization skills to determine if Fandango's ratings in 2015 had a bias towards rating movies better to sell more tickets.**
# 
# ---
# ---
# 
# **Complete the tasks written in bold.**
# 
# ---
# ----
# 
# ## Part One: Understanding the Background and Data
# 
# 
# **TASK: Read this article: [Be Suspicious Of Online Movie Ratings, Especially Fandango’s](http://fivethirtyeight.com/features/fandango-movies-ratings/)**

# ----
# 
# **TASK: After reading the article, read these two tables giving an overview of the two .csv files we will be working with:**
# 
# ### The Data
# 
# This is the data behind the story [Be Suspicious Of Online Movie Ratings, Especially Fandango’s](http://fivethirtyeight.com/features/fandango-movies-ratings/) openly available on 538's github: https://github.com/fivethirtyeight/data. There are two csv files, one with Fandango Stars and Displayed Ratings, and the other with aggregate data for movie ratings from other sites, like Metacritic,IMDB, and Rotten Tomatoes.
# 
# #### all_sites_scores.csv

# -----
# 
# `all_sites_scores.csv` contains every film that has a Rotten Tomatoes rating, a RT User rating, a Metacritic score, a Metacritic User score, and IMDb score, and at least 30 fan reviews on Fandango. The data from Fandango was pulled on Aug. 24, 2015.

# Column | Definition
# --- | -----------
# FILM | The film in question
# RottenTomatoes | The Rotten Tomatoes Tomatometer score  for the film
# RottenTomatoes_User | The Rotten Tomatoes user score for the film
# Metacritic | The Metacritic critic score for the film
# Metacritic_User | The Metacritic user score for the film
# IMDB | The IMDb user score for the film
# Metacritic_user_vote_count | The number of user votes the film had on Metacritic
# IMDB_user_vote_count | The number of user votes the film had on IMDb

# ----
# ----
# 
# #### fandango_scape.csv

# `fandango_scrape.csv` contains every film 538 pulled from Fandango.
# 
# Column | Definiton
# --- | ---------
# FILM | The movie
# STARS | Number of stars presented on Fandango.com
# RATING |  The Fandango ratingValue for the film, as pulled from the HTML of each page. This is the actual average score the movie obtained.
# VOTES | number of people who had reviewed the film at the time we pulled it.

# ----
# 
# **TASK: Import any libraries you think you will use:**

# In[2]:


# IMPORT HERE!


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Part Two: Exploring Fandango Displayed Scores versus True User Ratings
# 
# Let's first explore the Fandango ratings to see if our analysis agrees with the article's conclusion.
# 
# **TASK: Run the cell below to read in the fandango_scrape.csv file**

# In[2]:


fandango = pd.read_csv("fandango_scrape.csv")


# **TASK: Explore the DataFrame Properties and Head.**

# In[3]:


fandango.head()


# In[4]:


fandango.info()


# In[5]:


fandango.head()


# In[7]:


col_names = ['STARS', 'RATING', 'VOTES']

functions = {
    np.mean: 'Mean',
    np.std: 'Std',
    np.amin: 'Min',
    np.amax: 'Max',
    'count': 'Count',
    25: '25%',
    50: '50%',
    75: '75%'
}

# Create an empty DataFrame
result_df = pd.DataFrame(index=functions.values(), columns=col_names)

for col in col_names:
    for func, name in functions.items():
        if func == 'count':
            result = fandango[col].count()
        elif isinstance(func, int):
            result = np.percentile(fandango[col], func)
        else:
            result = func(fandango[col])
            if isinstance(result, float):
                result = round(result, 2)  # Round floating-point numbers to 2 decimal places
        result_df.at[name, col] = result
result_df


# **TASK: Let's explore the relationship between popularity of a film and its rating. Create a scatterplot showing the relationship between rating and votes. Feel free to edit visual styling to your preference.**

# In[9]:


# CODE HERE


# In[8]:


Rating = fandango.iloc[: , 2]
Votes = fandango.iloc[: , 3]


# In[9]:


plt.figure(figsize=(12, 6)) 
plt.scatter(Rating, Votes, color = 'Blue', s=100)
plt.xlabel('RATING')
plt.ylabel('VOTES')


# **TASK: Calculate the correlation between the columns:**

# In[12]:


# CODE HERE


# In[10]:


correlation_matrix = fandango.corr()
correlation_matrix


# In[ ]:





# **TASK: Assuming that every row in the FILM title column has the same format:**
# 
#     Film Title Name (Year)
#     
# **Create a new column that is able to strip the year from the title strings and set this new column as YEAR**

# In[ ]:


# CODE HERE


# In[11]:



fandango = fandango.copy()

# Extract the year from the 'FILM' column using .str.extract and .loc
fandango.loc[:, 'YEAR'] = fandango['FILM'].str.extract(r'\((\d{4})\)', expand=False)

# Display the first few rows to verify
fandango.head()


# **TASK: How many movies are in the Fandango DataFrame per year?**

# In[ ]:


#CODE HERE


# In[12]:


movie_per_year = fandango['YEAR'].value_counts()
movie_per_year


# In[ ]:





# **TASK: Visualize the count of movies per year with a plot:**

# In[ ]:


#CODE HERE


# In[13]:


movie_per_year = fandango['YEAR'].value_counts().reset_index()

movies_per_year_array = movie_per_year.to_numpy()

#making the plot
plt.bar(movies_per_year_array[: , 0], movies_per_year_array[:, -1], color= ['Blue','orange', 'green', 'red', 'Blue'])
plt.xlabel('YEAR')
plt.ylabel('count')


# In[ ]:





# **TASK: What are the 10 movies with the highest number of votes?**

# In[14]:


#CODE HERE
fandango.sort_values(by = 'VOTES', ascending=False).head(10)


# In[ ]:





# **TASK: How many movies have zero votes?**

# In[15]:


#CODE HERE
Movies_with_zero_votes = (fandango['VOTES']==0).sum()
Movies_with_zero_votes


# In[ ]:





# **TASK: Create DataFrame of only reviewed films by removing any films that have zero votes.**

# In[ ]:


#CODE HERE


# In[16]:


fandango = fandango[fandango['VOTES'] != 0]
fandango


# ----
# 
# **As noted in the article, due to HTML and star rating displays, the true user rating may be slightly different than the rating shown to a user. Let's visualize this difference in distributions.**
# 
# **TASK: Create a KDE plot (or multiple kdeplots) that displays the distribution of ratings that are displayed (STARS) versus what the true rating was from votes (RATING). Clip the KDEs to 0-5.**

# In[ ]:


#CODE HERE


# In[17]:


plt.figure(figsize=(14, 7))  # Adjust figure size if needed

sns.kdeplot(fandango['RATING'], label='True Rating',    shade=True )
sns.kdeplot(fandango['STARS'], label='Stars Displayed', shade=True)
plt.xlim(0,5)
plt.ylim(0, 0.6)

# Add labels and title
plt.xlabel('Rating')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()


# In[ ]:





# **TASK: Let's now actually quantify this discrepancy. Create a new column of the different between STARS displayed versus true RATING. Calculate this difference with STARS-RATING and round these differences to the nearest decimal point.**

# In[ ]:


#CODE HERE


# In[18]:


pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings

fandango['STARS_DIFF'] = fandango['STARS'] - fandango['RATING']
fandango


# In[ ]:





# **TASK: Create a count plot to display the number of times a certain difference occurs:**

# In[214]:


#CODE HERE

# Create the count plot
sns.countplot(data=fandango, x='STARS_DIFF' )

# Show the plot
plt.show()


# In[ ]:





# **TASK: We can see from the plot that one movie was displaying over a 1 star difference than its true rating! What movie had this close to 1 star differential?**

# In[20]:


#CODE HERE
# Assuming 'fandango' is your DataFrame containing the necessary columns
Desired = fandango[(fandango['STARS_DIFF'] >= 0.9) & (fandango['STARS_DIFF'] <= 1.1)]
Desired


# In[ ]:





# ## Part Three: Comparison of Fandango Ratings to Other Sites
# 
# Let's now compare the scores from Fandango to other movies sites and see how they compare.
# 
# **TASK: Read in the "all_sites_scores.csv" file by running the cell below**

# In[21]:


all_sites = pd.read_csv("all_sites_scores.csv")


# **TASK: Explore the DataFrame columns, info, description.**

# In[22]:


all_sites.head()


# In[ ]:





# In[23]:


all_sites.info()


# In[ ]:





# In[24]:


col_names = ['RottenTomatoes', 'RottenTomatoes_User', 'Metacritic', 'Metacritic_User', 'IMDB' , 'Metacritic_user_vote_count','IMDB_user_vote_count']

functions = {
    'count': 'Count',
    np.mean: 'Mean',
    np.std: 'Std',
    np.amin: 'Min',
    25: '25%',
    50: '50%',
    75: '75%',
    np.amax: 'Max'
}

# Create an empty DataFrame
result_df = pd.DataFrame(index=functions.values(), columns=col_names)

for col in col_names:
    for func, name in functions.items():
        if func == 'count':
            result = all_sites[col].count()
        elif isinstance(func, int):
            result = np.percentile(all_sites[col], func)
        else:
            result = func(all_sites[col])
            if isinstance(result, float):
                result  # Round floating-point numbers to 2 decimal places
        result_df.at[name, col] = result

result_df


# In[ ]:





# ### Rotten Tomatoes
# 
# Let's first take a look at Rotten Tomatoes. RT has two sets of reviews, their critics reviews (ratings published by official critics) and user reviews.
# 
# **TASK: Create a scatterplot exploring the relationship between RT Critic reviews and RT User reviews.**

# In[25]:


# CODE HERE
plt.figure(figsize=(14, 7))  # Adjust figure size if needed

plt.scatter(x = all_sites['RottenTomatoes'], y = all_sites['RottenTomatoes_User'])
plt.ylim(0,100)
plt.xlim(0,100)
plt.xlabel('RottenTomatoes')
plt.ylabel('RottenTomatoes_User')


# In[ ]:





# Let's quantify this difference by comparing the critics ratings and the RT User ratings. We will calculate this with RottenTomatoes-RottenTomatoes_User. Note: Rotten_Diff here is Critics - User Score. So values closer to 0 means aggrement between Critics and Users. Larger positive values means critics rated much higher than users. Larger negative values means users rated much higher than critics.
# 
# **TASK: Create a new column based off the difference between critics ratings and users ratings for Rotten Tomatoes. Calculate this with RottenTomatoes-RottenTomatoes_User**

# In[26]:


#CODE HERE
all_sites['ROTTEN DIFF']= all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']
all_sites.head()


# In[ ]:





# Let's now compare the overall mean difference. Since we're dealing with differences that could be negative or positive, first take the absolute value of all the differences, then take the mean. This would report back on average to absolute difference between the critics rating versus the user rating.

# **TASK: Calculate the Mean Absolute Difference between RT scores and RT User scores as described above.**

# In[27]:


# CODE HERE
abs(all_sites['ROTTEN DIFF']).mean()


# In[ ]:





# **TASK: Plot the distribution of the differences between RT Critics Score and RT User Score. There should be negative values in this distribution plot. Feel free to use KDE or Histograms to display this distribution.**

# In[121]:


#CODE HERE
plt.figure(figsize=(14, 7))
plt.grid(False)
sns.histplot(all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User'], kde =True, bins=21, color='skyblue', edgecolor='black')
plt.xlabel('ROTTEN DIFF')
plt.ylabel('Count')
plt.title('Abs Difference between R Critics Score and RT User Score')
plt.show()



# In[ ]:





# **TASK: Now create a distribution showing the *absolute value* difference between Critics and Users on Rotten Tomatoes.**

# In[120]:


#CODE HERE
plt.figure(figsize=(14, 7))
plt.grid(False)
sns.histplot(abs(all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']), kde=True, bins=21, color='skyblue', edgecolor='black')
plt.xlabel('ROTTEN DIFF')
plt.ylabel('Count')
plt.title('Abs Difference between R Critics Score and RT User Score')
plt.show()


# In[ ]:





# **Let's find out which movies are causing the largest differences. First, show the top 5 movies with the largest *negative* difference between Users and RT critics. Since we calculated the difference as Critics Rating - Users Rating, then large negative values imply the users rated the movie much higher on average than the critics did.**

# **TASK: What are the top 5 movies users rated higher than critics on average:**

# In[79]:


# CODE HERE
result = all_sites.sort_values(by='ROTTEN DIFF', ascending=True)[['FILM', 'ROTTEN DIFF']]
result.head(6)


# In[ ]:





# **TASK: Now show the top 5 movies critics scores higher than users on average.**

# In[80]:


# CODE HERE
result = all_sites.sort_values(by='ROTTEN DIFF', ascending=False)[['FILM', 'ROTTEN DIFF']]
result.head(6)


# In[ ]:





# ## MetaCritic
# 
# Now let's take a quick look at the ratings from MetaCritic. Metacritic also shows an average user rating versus their official displayed rating.

# **TASK: Display a scatterplot of the Metacritic Rating versus the Metacritic User rating.**

# In[101]:


all_sites.head()


# In[106]:


# CODE HERE
plt.figure(figsize=(14,7)) 
plt.grid(False) 
plt.xlim(0,100)
plt.ylim(0,10)

plt.scatter(all_sites['Metacritic'], all_sites['Metacritic_User'], color = 'Blue',edgecolor='black', s=70)
plt.xlabel('Metacritic_user_vote_count')
plt.ylabel('IMDB_user_vote_count')


# In[ ]:





# ## IMDB
# 
# Finally let's explore IMDB. Notice that both Metacritic and IMDB report back vote counts. Let's analyze the most popular movies.
# 
# **TASK: Create a scatterplot for the relationship between vote counts on MetaCritic versus vote counts on IMDB.**

# In[107]:


#CODE HERE
plt.figure(figsize=(14,7)) 
plt.grid(False) 

plt.scatter(all_sites['Metacritic_user_vote_count'], all_sites['IMDB_user_vote_count'], color = 'Blue',edgecolor='black', s=70)
plt.xlabel('Metacritic_user_vote_count')
plt.ylabel('IMDB_user_vote_count')


# In[ ]:





# **Notice there are two outliers here. The movie with the highest vote count on IMDB only has about 500 Metacritic ratings. What is this movie?**
# 
# **TASK: What movie has the highest IMDB user vote count?**

# In[122]:


#CODE HERE
IMDB_Best_Result = all_sites.sort_values(by= 'IMDB_user_vote_count',    ascending=False ).head(1)
IMDB_Best_Result


# In[ ]:





# **TASK: What movie has the highest Metacritic User Vote count?**

# In[125]:


#CODE HERE
Metacritic_Best_Result = all_sites.sort_values(by= 'Metacritic_user_vote_count',    ascending=False ).head(1)
Metacritic_Best_Result


# In[ ]:





# ## Fandago Scores vs. All Sites
# 
# Finally let's begin to explore whether or not Fandango artificially displays higher ratings than warranted to boost ticket sales.

# **TASK: Combine the Fandango Table with the All Sites table. Not every movie in the Fandango table is in the All Sites table, since some Fandango movies have very little or no reviews. We only want to compare movies that are in both DataFrames, so do an *inner* merge to merge together both DataFrames based on the FILM columns.**

# In[150]:


#CODE HERE
original_fandango = fandango.drop('STARS_DIFF'  , axis=1)
merged_df = pd.merge(original_fandango, all_sites, on='FILM', how='inner')


# In[151]:


merged_df.info()


# In[ ]:





# In[194]:


merged_df.head()


# In[ ]:





# ### Normalize columns to Fandango STARS and RATINGS 0-5
# 
# Notice that RT,Metacritic, and IMDB don't use a score between 0-5 stars like Fandango does. In order to do a fair comparison, we need to *normalize* these values so they all fall between 0-5 stars and the relationship between reviews stays the same.
# 
# **TASK: Create new normalized columns for all ratings so they match up within the 0-5 star range shown on Fandango. There are many ways to do this.**
# 
# Hint link: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
# 
# 
# Easier Hint:
# 
# Keep in mind, a simple way to convert ratings:
# * 100/20 = 5
# * 10/2 = 5

# In[ ]:


# CODE HERE


# In[169]:


#min & max of Rotten Tomatoes
# rt_min = merged_df['RottenTomatoes'].min()
# rt_max = merged_df['RottenTomatoes'].max()
rt_min = 0
rt_max = 100

#normalizing
merged_df['RT_Norm'] = (5 * ((merged_df['RottenTomatoes'] - rt_min) / (rt_max - rt_min))).round(1)


# In[170]:



#min & max of Rotten Tomatoes users
# rtu_min = merged_df['RottenTomatoes_User'].min()
# rtu_max = merged_df['RottenTomatoes_User'].max()
rtu_min = 0
rtu_max = 100

#normalizing
merged_df['RTU_Norm'] = (5 * ((merged_df['RottenTomatoes_User'] - rtu_min) / (rtu_max - rtu_min))).round(1)


# In[171]:


#min & max of metacritic
# metacritic_min = merged_df['Metacritic'].min()
# metacritic_max = merged_df['Metacritic'].max()
metacritic_min = 0
metacritic_max = 100
#normalizing
merged_df['Meta_Norm'] = (5 * ((merged_df['Metacritic'] - metacritic_min) / (metacritic_max - metacritic_min))).round(1)


# In[195]:


#min & max of metacritic users
# metacritic_U_min = merged_df['Metacritic_User'].min()
# metacritic_U_max = merged_df['Metacritic_User'].max()

metacritic_U_min = 0
metacritic_U_max = 10
#normalizing
merged_df['Meta_U_Norm'] = (5 * ((merged_df['Metacritic_User'] - metacritic_U_min) / (metacritic_U_max - metacritic_U_min))).round(1)


# In[173]:


#min & max of IMDB
# imdb_min = merged_df['IMDB'].min()
# imdb_max = merged_df['IMDB'].max()
imdb_min = 0
imdb_max = 10
#normalizing
merged_df['IMDB_Norm'] = (5 * ((merged_df['IMDB'] - imdb_min) / (imdb_max - imdb_min))).round(1)


# In[196]:


merged_df.head()


# In[ ]:





# **TASK: Now create a norm_scores DataFrame that only contains the normalizes ratings. Include both STARS and RATING from the original Fandango table.**

# In[ ]:


#CODE HERE


# In[197]:


merged_df_2 = merged_df[['STARS', 'RATING','RT_Norm', 'RTU_Norm', 'Meta_Norm', 'Meta_U_Norm', 'IMDB_Norm' ]]
merged_df_2.head()


# In[ ]:





# ### Comparing Distribution of Scores Across Sites
# 
# 
# Now the moment of truth! Does Fandango display abnormally high ratings? We already know it pushs displayed RATING higher than STARS, but are the ratings themselves higher than average?
# 
# 
# **TASK: Create a plot comparing the distributions of normalized ratings across all sites. There are many ways to do this, but explore the Seaborn KDEplot docs for some simple ways to quickly show this. Don't worry if your plot format does not look exactly the same as ours, as long as the differences in distribution are clear.**
# 
# Quick Note if you have issues moving the legend for a seaborn kdeplot: https://github.com/mwaskom/seaborn/issues/2280

# In[200]:


#CODE HERE
plt.figure(figsize=(16,8))
plt.grid(False)
sns.kdeplot(x = merged_df_2['STARS']  ,shade=True, color = 'red', label= 'STARS' )
sns.kdeplot(x = merged_df_2['RATING']  ,shade=True, color = 'Blue', label= 'RATING' )
sns.kdeplot(x = merged_df_2['RT_Norm']  ,shade=True, color = 'Green', label= 'RT_Norm' )
sns.kdeplot(x = merged_df_2['RTU_Norm']  ,shade=True, color = 'Purple', label= 'RTU_Norm' )
sns.kdeplot(x = merged_df_2['Meta_Norm']  ,shade=True, color = 'orange', label= 'Meta_Norm' )
sns.kdeplot(x = merged_df_2['Meta_U_Norm']  ,shade=True, color = 'yellow', label= 'Meta_U_Norm' )
sns.kdeplot(x = merged_df_2['IMDB_Norm']  ,shade=True, color = 'brown', label= 'IMDB_Norm' )




plt.xlim(0,5)
plt.ylabel('Density')
plt.legend(loc= 'upper left')


# In[ ]:





# **Clearly Fandango has an uneven distribution. We can also see that RT critics have the most uniform distribution. Let's directly compare these two.**
# 
# **TASK: Create a KDE plot that compare the distribution of RT critic ratings against the STARS displayed by Fandango.**

# In[208]:


#CODE HERE
plt.figure(figsize=(16,8))
plt.grid(False)
sns.kdeplot(x = merged_df_2['RT_Norm']  ,shade=True, color = 'Red', label= 'RT_Norm' )

sns.kdeplot(x = merged_df_2['STARS']  ,shade=True, color = 'Blue', label= 'STARS' )

plt.xlim(0,5)

plt.ylabel('Density')
plt.legend(loc= 'upper left')


# In[ ]:





# **TASK: Create a histplot comparing all normalized scores.**

# In[215]:


merged_df_2.head()


# In[250]:


#CODE HERE
plt.figure(figsize=(16,8))
plt.grid(False)
plt.xticks(np.arange(0, 6, 1))


sns.histplot(data=merged_df_2['STARS'], bins=30, color='Blue', label='STARS' , linewidth=0, edgecolor='black')
sns.histplot(data=merged_df_2['RATING'], bins=30, color='Orange', label='RATING', linewidth=0, edgecolor='black')
sns.histplot(data=merged_df_2['RT_Norm'], bins=30, color='Green', label='RT_Norm', linewidth=0, edgecolor='black')
sns.histplot(data=merged_df_2['RTU_Norm'], bins=30, color='Red', label='RTU_Norm', linewidth=0, edgecolor='black')
sns.histplot(data=merged_df_2['Meta_Norm'], bins=30, color='purple', label='Meta_Norm', linewidth=0, edgecolor='black')
sns.histplot(data=merged_df_2['Meta_U_Norm'], bins=30, color='Brown', label='Meta_U_Norm', linewidth=0, edgecolor='black')
sns.histplot(data=merged_df_2['IMDB_Norm'], bins=30, color='Pink', label='IMDB_Norm', linewidth=0, edgecolor='black')


plt.ylabel('Count')
plt.legend(loc = 'upper right')
plt.show()


# In[ ]:





# 
# ### How are the worst movies rated across all platforms?
# 
# **TASK: Create a clustermap visualization of all normalized scores. Note the differences in ratings, highly rated movies should be clustered together versus poorly rated movies. Note: This clustermap does not need to have the FILM titles as the index, feel free to drop it for the clustermap.**

# In[253]:


# CODE HERE
merged_df_2 = merged_df_2.drop('FILM', axis=1, errors='ignore')

# Create a clustermap of the normalized scores
sns.clustermap(merged_df_2, cmap='viridis', figsize=(12, 10))

plt.title('Clustermap of Normalized Scores')
plt.show()


# In[ ]:





# **TASK: Clearly Fandango is rating movies much higher than other sites, especially considering that it is then displaying a rounded up version of the rating. Let's examine the top 10 worst movies. Based off the Rotten Tomatoes Critic Ratings, what are the top 10 lowest rated movies? What are the normalized scores across all platforms for these movies? You may need to add the FILM column back in to your DataFrame of normalized scores to see the results.**

# In[275]:


# CODE HERE
merged_df_2['FILM']= merged_df['FILM']
ten_worst =merged_df_2.sort_values(by = 'RT_Norm',ascending=True ).head(10)
ten_worst


# In[ ]:





# In[ ]:





# **FINAL TASK: Visualize the distribution of ratings across all sites for the top 10 worst movies.**

# In[266]:


# CODE HERE
plt.figure(figsize= (16, 8))
plt.grid(False)
sns.kdeplot(ten_worst['STARS'],     shade=True, color = 'Red', label = 'STARS')
sns.kdeplot(ten_worst['RATING'],     shade=True, color = 'Blue' , label= 'RATING')
sns.kdeplot(ten_worst['RT_Norm'],     shade=True, color = 'Green' , label= 'RT_Norm')
sns.kdeplot(ten_worst['RTU_Norm'],     shade=True, color = 'Purple' , label= 'RTU_Norm')
sns.kdeplot(ten_worst['Meta_Norm'],     shade=True, color = 'Orange' , label= 'Meta_Norm')
sns.kdeplot(ten_worst['Meta_U_Norm'],     shade=True, color = 'Yellow' , label= 'Meta_U_Norm')
sns.kdeplot(ten_worst['IMDB_Norm'],     shade=True, color = 'Brown' , label= 'IMDB_Norm')
plt.title(' Rating for RT Critics 10 worst reviewed FILMS')
plt.legend(loc = 'upper right')


# In[ ]:





# ---
# ----
# 
# <img src="https://upload.wikimedia.org/wikipedia/en/6/6f/Taken_3_poster.jpg">
# 
# **Final thoughts: Wow! Fandango is showing around 3-4 star ratings for films that are clearly bad! Notice the biggest offender, [Taken 3!](https://www.youtube.com/watch?v=tJrfImRCHJ0). Fandango is displaying 4.5 stars on their site for a film with an [average rating of 1.86](https://en.wikipedia.org/wiki/Taken_3#Critical_response) across the other platforms!**

# In[277]:


print(ten_worst.iloc[1, :])


# In[ ]:





# In[ ]:


0.4+2.3+1.3+2.3+3


# In[ ]:


9.3/5


# ----
