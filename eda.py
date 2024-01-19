import pandas as pd
import json
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#en cok likelananlara bakılabilir-> bunu yapmak için like_data['likes_media_likes'] içindeki title ve string_list_data içindeki timestamp kullanılabilir,
#->distinction between accounts


# Load the JSON file
with open(r'insta_data\your_instagram_activity\likes\liked_posts.json') as file:
    like_data = json.load(file)
with open(r'insta_data\your_instagram_activity\saved\saved_posts.json') as file:
    save_data = json.load(file)

# Extract the relevant information
liked_posts = []
saved_posts = []

limitToTerm = False
termStart= datetime(2023, 8, 31)

for post in like_data['likes_media_likes']:
    account = post.get('title', None)
    for data in post['string_list_data']:
        timestamp = datetime.fromtimestamp(data['timestamp'])
        if not limitToTerm or (timestamp > termStart):
            liked_posts.append({'account': account, 'date': timestamp})

for post in save_data['saved_saved_media']:
    account = post.get('title', None)
    data = post['string_map_data']['Saved on']
    timestamp = datetime.fromtimestamp(data['timestamp'])
    if not limitToTerm or (timestamp > termStart):
        saved_posts.append({'account': account, 'date': timestamp})

# Create a DataFrame
df_likes = pd.DataFrame(liked_posts)
df_saves = pd.DataFrame(saved_posts)

# Delete rows with missing account field
df_likes = df_likes.dropna(subset=['account'])
df_saves = df_saves.dropna(subset=['account'])

# Group by account and sum the number of likes
likes_per_account = df_likes.groupby('account')['date'].count().reset_index()

# Sort the DataFrame by the number of likes in descending order
likes_per_account = likes_per_account.sort_values('date', ascending=False)
likes_per_account = likes_per_account.rename(columns={'date': 'likes'})
# Print the top 10 most liked accounts
top_liked_accounts = likes_per_account.head(10)

# Plot the top 10 most liked accounts in a bar graph
plt.bar(top_liked_accounts['account'], top_liked_accounts['likes'])
plt.xlabel('Account')
plt.ylabel('Number of Likes')
plt.title('Top 10 Most Liked Accounts')
plt.show()


# Convert the 'date' column to datetime
df_likes['date'] = pd.to_datetime(df_likes['date'])
df_saves['date'] = pd.to_datetime(df_saves['date'])

# Group by day and count the number of likes/saves
likes_per_day = df_likes.groupby(df_likes['date'].dt.date).size().reset_index(name='likes')
likes_per_day['date'] = pd.to_datetime(likes_per_day['date']) - pd.to_timedelta(7, unit='d')
likes_per_week = likes_per_day.groupby([pd.Grouper(key='date', freq='W')])['likes'].sum().to_frame().reset_index()

saves_per_day = df_saves.groupby(df_saves['date'].dt.date).size().reset_index(name='saves')
saves_per_day['date'] = pd.to_datetime(saves_per_day['date']) - pd.to_timedelta(7, unit='d')
saves_per_week = saves_per_day.groupby([pd.Grouper(key='date', freq='W')])['saves'].sum().to_frame().reset_index()

# Merge the likes_per_week and saves_per_week DataFrames
df_total = pd.merge(likes_per_week, saves_per_week, on='date', how='outer')
df_total = df_total.fillna(0)
df_total['total'] = df_total['likes'] + df_total['saves']

# Plot the likes/saves per week as a line plot
plt.plot(likes_per_week['date'], likes_per_week['likes'], label='Likes')
plt.plot(saves_per_week['date'], saves_per_week['saves'], label='Saves')
plt.plot(df_total['date'], df_total['total'], label='Total')

plt.xlabel('Week')
plt.ylabel('Value')
plt.title('Likes and saves per Week')
plt.legend()

plt.show()
