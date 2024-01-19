#second hyp could be i prioritize some interactions more
#reply rate through time
import pandas as pd
import json
import numpy as np
from datetime import datetime
import seaborn as sns
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

limitToTerm = False
termStart= datetime(2023, 8, 31)

DM_file_names=[r'insta_data\your_instagram_activity\messages\inbox\ardabaristonbil_933043168094634\message_1.json',
           r'insta_data\your_instagram_activity\messages\inbox\aylinozkaya_496071265125162\message_1.json',
           r'insta_data\your_instagram_activity\messages\inbox\mertpolat_496349915097297\message_1.json',
           r'insta_data\your_instagram_activity\messages\inbox\taha_813439610054991\message_1.json',
           r'insta_data\your_instagram_activity\messages\inbox\utku_906578497407768\message_1.json']

group_file_names=[r'insta_data\your_instagram_activity\messages\inbox\bucalismaz_5324286234349409\message_1.json',
               r'insta_data\your_instagram_activity\messages\inbox\adpafterlife_5939229612835128\message_1.json']

n_dms = len(DM_file_names)
n_groups = len(group_file_names)

DM_file_data = {}
group_file_data = {}
# Load the JSON files for DMs
for file_name in DM_file_names:
    with open(file_name) as file:
        DM_file_data[file_name] = json.load(file)

# Load the JSON files for group messages
for file_name in group_file_names:
    with open(file_name) as file:
        group_file_data[file_name] = json.load(file)



dm_dfs = []
for file_name in DM_file_names:
    messages=[]
    for message in DM_file_data[file_name]['messages']:
        if 'content' not in message:
            continue
        is_post = message['content'].endswith('attachment.')
        sent_by_me = message['sender_name'] == 'Ege'
        timestamp = datetime.fromtimestamp(message['timestamp_ms']/1000)
        reacted_by_me = False
        reacted_by_else = False
        if 'reactions' in message:
            for reaction in message['reactions']:
                if reaction['actor'] == 'Ege':
                    reacted_by_me = True
                else:
                    reacted_by_else = True
                if reacted_by_me and reacted_by_else:
                    break
        if not limitToTerm or (timestamp > termStart):
            messages.append({'sent_by_me': sent_by_me, 'timestamp': timestamp, 'is_post': is_post, 
                             'reacted_by_me': reacted_by_me, 'reacted_by_else': reacted_by_else})
    df = pd.DataFrame(messages)
    dm_dfs.append(df)

group_dfs = []
for file_name in group_file_names:
    messages=[]
    for message in group_file_data[file_name]['messages']:
        if 'content' not in message:
            continue
        is_post = message['content'].endswith('attachment.')
        sent_by_me = message['sender_name'] == 'Ege'
        timestamp = datetime.fromtimestamp(message['timestamp_ms']/1000)
        reacted_by_me = False
        reacted_by_else = False
        if 'reactions' in message:
            for reaction in message['reactions']:
                if reaction['actor'] == 'Ege':
                    reacted_by_me = True
                else:
                    reacted_by_else = True
                if reacted_by_me and reacted_by_else:
                    break
        if not limitToTerm or (timestamp > termStart):
            messages.append({'sent_by_me': sent_by_me, 'timestamp': timestamp, 'is_post': is_post,
                              'reacted_by_me': reacted_by_me, 'reacted_by_else': reacted_by_else})
    df = pd.DataFrame(messages)
    group_dfs.append(df)

incoming_reaction_rates = []
outgoing_reaction_rates = []
for df in dm_dfs:
    incoming_msg_df = df[(df['sent_by_me'] == False) & (df['is_post'] == True)]
    outgoing_msg_df = df[(df['sent_by_me'] == True) & (df['is_post'] == True)]

    incoming_reaction_rates.append(incoming_msg_df['reacted_by_me'].value_counts())
    outgoing_reaction_rates.append(outgoing_msg_df['reacted_by_else'].value_counts())

for df in group_dfs:
    incoming_msg_df = df[(df['sent_by_me'] == False) & (df['is_post'] == True)]
    outgoing_msg_df = df[(df['sent_by_me'] == True) & (df['is_post'] == True)]

    incoming_reaction_rates.append(incoming_msg_df['reacted_by_me'].value_counts())
    outgoing_reaction_rates.append(outgoing_msg_df['reacted_by_else'].value_counts())


incoming_df = pd.DataFrame(incoming_reaction_rates)
outgoing_df = pd.DataFrame(outgoing_reaction_rates)

# Reorder the columns
incoming_df = incoming_df.reindex([True, False], axis=1)
outgoing_df = outgoing_df.reindex([True, False], axis=1)


############################################################ plot p2p dms ############################################################
sns.set(style="whitegrid")

incoming_df.head(n_dms).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Account Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_dms), range((n_dms)), rotation=0)

# Title of plot
plt.title('Reaction rate of me to incoming DMs from people')
plt.show()

outgoing_df.head(n_dms).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Account Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_dms), range((n_dms)), rotation=0)

# Title of plot
plt.title('Reaction rate of others to outgoing DMs from me')
plt.show()

############################################################ plot group dms ############################################################
# Plot the data of group dms
sns.set(style="whitegrid")

incoming_df.tail(n_groups).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Group Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_groups), range((n_groups)), rotation=0)

# Title of plot
plt.title('Reaction rate of me to incoming DMs in groups')
plt.show()

outgoing_df.tail(n_groups).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Group Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_groups), range((n_groups)), rotation=0)

# Title of plot
plt.title('Reaction rate of others to outgoing DMs in groups')
plt.show()
