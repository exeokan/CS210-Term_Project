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

limitToTerm = True
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


# Create a dataframe for each DM
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

# calculate reaction rates
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

'''
############################################################ plot p2p dms ############################################################
sns.set(style="whitegrid")

incoming_df.head(n_dms).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Account Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_dms), range((n_dms)), rotation=0)
plt.legend(['Reacted', 'Not reacted'])
# Title of plot
plt.title('Reaction rate of me to incoming DMs from people')
plt.show()

outgoing_df.head(n_dms).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Account Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_dms), range((n_dms)), rotation=0)
plt.legend(['Reacted', 'Not reacted'])
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
plt.legend(['Reacted', 'Not reacted'])
# Title of plot
plt.title('Reaction rate of me to incoming DMs in groups')
plt.show()

outgoing_df.tail(n_groups).plot(kind='bar', stacked=True, color=['#669bbc', '#003049'])
plt.xlabel('Group Index')
plt.ylabel('Value')

# Set the x-axis tick labels to index numbers
plt.xticks(range(n_groups), range((n_groups)), rotation=0)
plt.legend(['Reacted', 'Not reacted'])
# Title of plot
plt.title('Reaction rate of others to outgoing DMs in groups')
plt.show()
'''
############################################################ reply rate through time ############################################################

reaction_date_dfs = []

all_dfs = dm_dfs + group_dfs


for df in dm_dfs:
    grouped_df = df[(df['sent_by_me'] == False) & (df['is_post'] == True)].groupby(df['timestamp'].dt.date)
    reaction_date_df = pd.DataFrame({
        'date': grouped_df['timestamp'].first().dt.date,
        'total_posts': grouped_df.size(),
        'posts_reacted_by_me': grouped_df['reacted_by_me'].sum()
    })
    
    reaction_date_df.reset_index()
    reaction_date_df['date'] = pd.to_datetime(reaction_date_df['date']) - pd.to_timedelta(7, unit='d')
    reaction_date_df = reaction_date_df.groupby([pd.Grouper(key='date', freq='W')])[['total_posts', 'posts_reacted_by_me']].sum().reset_index()

    reaction_date_df['reaction_ratio'] = reaction_date_df['posts_reacted_by_me'] / reaction_date_df['total_posts']
    reaction_date_dfs.append(reaction_date_df)

result_df = pd.DataFrame()
result_df['date'] = reaction_date_dfs[0]['date']
result_df['total_posts'] = 0
result_df['posts_reacted_by_me'] = 0

# Plot the data
for i in range(n_dms):
    plt.plot(reaction_date_dfs[i]['date'], reaction_date_dfs[i]['reaction_ratio'], label= i)
    print(reaction_date_dfs[i].head())
    result_df['total_posts'] += reaction_date_dfs[i]['total_posts']
    result_df['posts_reacted_by_me'] += reaction_date_dfs[i]['posts_reacted_by_me']
    print(result_df.head())

result_df['reaction_ratio'] = result_df['posts_reacted_by_me'] / result_df['total_posts']


plt.xlabel('Week')
plt.ylabel('Reaction Ratio')
plt.title('Reaction ratio of me to incoming DMs from people')
plt.legend()

plt.show()

#total 

plt.plot(result_df['date'], result_df['reaction_ratio'], label= 'total')
plt.xlabel('Week')
plt.ylabel('Reaction Ratio')
plt.title('Reaction ratio of me to TOTAL incoming DMs from people')
plt.legend()

plt.show()

#for groups
for i in range(n_groups):
    plt.plot(reaction_date_dfs[i]['date'], reaction_date_dfs[i]['reaction_ratio'], label= i)

plt.xlabel('Week')
plt.ylabel('Reaction Ratio')
plt.title('Reaction ratio of me to incoming DMs in groups')
plt.legend()

plt.show()

