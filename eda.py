import pandas as pd
import json
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Load the JSON file
with open('insta_data\your_instagram_activity\likes\liked_posts.json') as file:
    data = json.load(file)

# Extract the relevant information
liked_posts = []

for post in data['likes_media_likes']:
    account = post.get('title', None)
    for data in post['string_list_data']:
        timestamp = datetime.fromtimestamp(data['timestamp'])
        if timestamp > datetime(2023, 8, 31):
            liked_posts.append({'account': account, 'date': timestamp})

# Create a DataFrame
df = pd.DataFrame(liked_posts)
# Delete rows with missing account field
df = df.dropna(subset=['account'])

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Group by day and count the number of likes
likes_per_day = df.groupby(df['date'].dt.date).size().reset_index(name='likes')
likes_per_day['date'] = pd.to_datetime(likes_per_day['date']) - pd.to_timedelta(7, unit='d')

print(likes_per_day.info())
likes_per_week = likes_per_day.groupby([pd.Grouper(key='date', freq='W')])['likes'].sum()
# Print the new DataFrame
print(likes_per_week)

# Plot the likes per week as a line plot
plt.plot(likes_per_week.index, likes_per_week.values)
plt.xlabel('Week')
plt.ylabel('Likes')
plt.title('Likes per Week')


plt.show()
