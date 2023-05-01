# #!/usr/bin/env python
# # coding: utf-8

# # ## Look at Data

# # In[25]:


# import gzip
# import csv

# # Open the compressed CSV file and create a CSV reader object
# with gzip.open('C:\\Users\\vraparth\\Coding\\Untitled Folder\\listings.csv.gz', 'rt', encoding='utf-8', newline='') as csvFile:
#     reader = csv.reader(csvFile)

#     # Print the header row
#     header_row = next(reader)
#     print(header_row)


# # ## Taking necessary data - Data Processing

# # In[29]:


# import pandas as pd

# # Read the csv file
# df1 = pd.read_csv("C:\\Users\\vraparth\\Coding\\Untitled Folder\\listings.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')

# # Select the desired features
# df2 = df1[['id', 'name', 'description', 'neighborhood_overview', 'picture_url', 'listing_url', 'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews', 'review_scores_rating', 'neighbourhood', 'neighbourhood_group_cleansed']]

# # Write the new DataFrame to a csv file
# df2.to_csv("new_listings.csv", index=False)


# # ## Data Cleaning

# # In[45]:


# import pandas as pd
# import re

# # Read the csv file
# df3 = pd.read_csv('C:\\Users\\vraparth\\Coding\\Untitled Folder\\new_listings.csv')

# # Remove special characters from selected columns
# regex = re.compile('[^a-zA-Z0-9\s]')
# df3['name'] = df3['name'].apply(lambda x: regex.sub('', str(x)))
# df3['description'] = df3['description'].apply(lambda x: regex.sub('', str(x)))
# df3['neighborhood_overview'] = df3['neighborhood_overview'].apply(lambda x: regex.sub('', str(x)))
# df3['neighbourhood_cleansed'] = df3['neighbourhood_cleansed'].apply(lambda x: regex.sub('', str(x)))
# df3['neighbourhood'] = df3['neighbourhood'].apply(lambda x: regex.sub('', str(x)))
# df3['neighbourhood_group_cleansed'] = df3['neighbourhood_group_cleansed'].apply(lambda x: regex.sub('', str(x)))

# # Remove null values
# #df = df.dropna()

# # Save cleaned dataframe to a new csv file
# #print(df.head(10))

# # Write the new DataFrame to a csv file
# df3.to_csv("cleaned_listings.csv", index=False)
# print('cleaned list file created')


# ## Data Processing
# #### - Making data to be compatible to apply Content Based Filter
# #### - converts the price column to numerical form
# #### - creates a feature matrix by concatenating the text and amenities features
# #### - normalizes the feature matrix, and saves the preprocessed data to a new CSV file.

# In[87]:


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the dataset
# df_listings = pd.read_csv('C:\\Users\\vraparth\\Coding\\Untitled Folder\\cleaned_listings.csv', nrows=20000)

# # Select the relevant columns
# df_features = df_listings[['id', 'description', 'name', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'review_scores_rating']]

# # Clean the text features
# df_features['price'] = df_features['price'].str.replace('$', '').str.replace(',', '').astype(float)
# df_features['amenities'] = df_features['amenities'].fillna('')
# df_features['amenities'] = df_features['amenities'].str.lower()
# df_features['property_type'] = df_features['property_type'].fillna('')
# df_features['property_type'] = df_features['property_type'].str.lower()
# df_features['room_type'] = df_features['room_type'].fillna('')
# df_features['room_type'] = df_features['room_type'].str.lower()

# # Create the TF-IDF matrix for the text features
# tfidf = TfidfVectorizer(stop_words='english')
# amenities_tfidf = tfidf.fit_transform(df_features['amenities']).toarray()
# property_type_tfidf = tfidf.fit_transform(df_features['property_type']).toarray()
# room_type_tfidf = tfidf.fit_transform(df_features['room_type']).toarray()

# # Create the feature matrix for the numerical features
# num_features = df_features[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'review_scores_rating']].fillna(0)

# # Combine the feature matrices
# feature_matrix = pd.concat([pd.DataFrame(amenities_tfidf), pd.DataFrame(property_type_tfidf), pd.DataFrame(room_type_tfidf), num_features.reset_index(drop=True)], axis=1)

# # Compute the cosine similarity matrix
# cosine_sim = cosine_similarity(feature_matrix)



# # Define a function to recommend similar listings
# def recommend_listings(listing_id, n=10):
#     # Get the index of the listing_id
#     idx = df_listings[df_listings['id'] == listing_id].index[0]
#     # Get the similarity scores of the listing with other listings
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     # Sort the listings based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     # Get the indices of the n most similar listings
#     sim_indices = [i[0] for i in sim_scores[1:n+1]]
#     # Return the n most similar listings
#     return df_listings.iloc[sim_indices][['id', 'name', 'description', 'property_type', 'room_type', 'accommodates']]
# list1 = recommend_listings(16246564)
# print(list1)


# In[128]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
#df_listings = pd.read_csv('C:\\Users\\vraparth\\Coding\\Untitled Folder\\cleaned_listings.csv', nrows=20000)
df_listings = pd.read_csv('C:\\Users\\vraparth\\Coding\\Product Recommendation System\\cleaned_listings.csv', nrows=20000)




# Select the relevant columns
df_features = df_listings[['id', 'description', 'name', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'review_scores_rating']]

# Clean the text features
df_features['price'] = df_features['price'].str.replace('$', '').str.replace(',', '').astype(float)
df_features['amenities'] = df_features['amenities'].fillna('')
df_features['amenities'] = df_features['amenities'].str.lower()
df_features['property_type'] = df_features['property_type'].fillna('')
df_features['property_type'] = df_features['property_type'].str.lower()
df_features['room_type'] = df_features['room_type'].fillna('')
df_features['room_type'] = df_features['room_type'].str.lower()
df_features['name'] = df_features['name'].fillna('')
df_features['name'] = df_features['name'].str.lower()
df_features['description'] = df_features['description'].fillna('')
df_features['description'] = df_features['description'].str.lower()

# create combined text feature to find the cosine similarity with the input text
#combined_text_feature = name + " " + description+ " "+ room_type + " "+ property_type + " "+ amenities

# Create the TF-IDF matrix for the text features
tfidf = TfidfVectorizer(stop_words='english')
amenities_tfidf = tfidf.fit_transform(df_features['amenities']).toarray()
property_type_tfidf = tfidf.fit_transform(df_features['property_type']).toarray()
room_type_tfidf = tfidf.fit_transform(df_features['room_type']).toarray()

# # Create the feature matrix for the numerical features
# num_features = df_features[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'review_scores_rating']].fillna(0)

# # Combine the feature matrices
# feature_matrix = pd.concat([pd.DataFrame(amenities_tfidf), pd.DataFrame(property_type_tfidf), pd.DataFrame(room_type_tfidf)], axis=1)

# # Compute the cosine similarity matrix
# cosine_sim = cosine_similarity(feature_matrix)

# Define the features to use for the recommendations
features = ['name', 'description', 'property_type', 'room_type']

# Create a new column 'combined_features' which combines the text from all the selected features
df_listings['combined_features'] = df_listings[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Define the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the vectorizer using the 'combined_features' column
feature_matrix = tfidf_vectorizer.fit_transform(df_listings['combined_features'])




# Convert the sparse matrix to a dataframe
feature_matrix_df = pd.DataFrame.sparse.from_spmatrix(feature_matrix)

# View the first 5 rows of the dataframe
#print(feature_matrix_df.head())


def recommend_listings_text(input_text, n=10):
    # Transform the input text using the same TfidfVectorizer used for the listing descriptions
    input_vec = tfidf_vectorizer.transform([input_text])
    
    # Compute the cosine similarity between the input vector and all the listings
    sim_scores = cosine_similarity(input_vec, feature_matrix)

    # Get the indices of the n most similar listings
    sim_indices = np.argsort(sim_scores.ravel())[-n:][::-1]

    # Return the n most similar listings
    return df_listings.iloc[sim_indices][['name', 'description','picture_url',  'price', 'room_type']]


#df_recommend = recommend_listings_text("2 bedroom with a swimming pool")
#print(df_recommend)



# In[ ]:




