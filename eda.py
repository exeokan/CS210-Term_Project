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
#likes_per_day = df.groupby(df['date'].dt.date).size().reset_index(name='likes')
likes_per_day = df.groupby(df['date'].dt.week).size().reset_index(name='likes')
print(df.head(30))
# Print the new DataFrame
print(likes_per_day)



# Create a pivot table to aggregate the likes by date and account
pivot_table = likes_per_day.pivot_table(index='likes', columns='date', aggfunc='size')

# Fill NaN values with 0
pivot_table = pivot_table.fillna(0)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlGnBu')
plt.title('Likes Distribution Over Time')
plt.xlabel('Date')
plt.ylabel('Likes')
plt.show()