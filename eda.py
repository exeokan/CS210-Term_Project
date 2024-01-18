import pandas as pd
import json
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Load the JSON file
with open('insta_data\your_instagram_activity\likes\liked_posts.json') as file:
    like_data = json.load(file)
with open('insta_data\your_instagram_activity\saved\saved_posts.json') as file:
    save_data = json.load(file)
# Extract the relevant information
liked_posts = []
saved_posts = []

for post in like_data['likes_media_likes']:
    account = post.get('title', None)
    for data in post['string_list_data']:
        timestamp = datetime.fromtimestamp(data['timestamp'])
        if timestamp > datetime(2023, 8, 31):
            liked_posts.append({'account': account, 'date': timestamp})

for post in save_data['saved_saved_media']:
    account = post.get('title', None)
    data = post['string_map_data']['Saved on']
    timestamp = datetime.fromtimestamp(data['timestamp'])
    if timestamp > datetime(2023, 8, 31):
        saved_posts.append({'account': account, 'date': timestamp})

# Create a DataFrame
df_likes = pd.DataFrame(liked_posts)
df_saves = pd.DataFrame(saved_posts)
# Delete rows with missing account field
df_likes = df_likes.dropna(subset=['account'])
df_saves = df_saves.dropna(subset=['account'])
# Convert the 'date' column to datetime
df_likes['date'] = pd.to_datetime(df_likes['date'])
df_saves['date'] = pd.to_datetime(df_saves['date'])

# Group by day and count the number of likes
likes_per_day = df_likes.groupby(df_likes['date'].dt.date).size().reset_index(name='likes')
likes_per_day['date'] = pd.to_datetime(likes_per_day['date']) - pd.to_timedelta(7, unit='d')

likes_per_week = likes_per_day.groupby([pd.Grouper(key='date', freq='W')])['likes'].sum().to_frame().reset_index()

saves_per_day = df_saves.groupby(df_saves['date'].dt.date).size().reset_index(name='saves')
saves_per_day['date'] = pd.to_datetime(saves_per_day['date']) - pd.to_timedelta(7, unit='d')
saves_per_week = saves_per_day.groupby([pd.Grouper(key='date', freq='W')])['saves'].sum().to_frame().reset_index()

# Print the new DataFrame
print(likes_per_week.head())
print(saves_per_week.head())
# Plot the likes per week as a line plot
plt.plot(likes_per_week['date'], likes_per_week['likes'], label='Likes')
plt.plot(saves_per_week['date'], saves_per_week['saves'], label='Saves')
plt.xlabel('Week')
plt.ylabel('Likes')
plt.title('Likes per Week')
plt.legend()

plt.show()
