import pandas as pd
import json
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import numpy as np

# Specify the file path
file_path = 'term_info.csv'

# Read the CSV file into a dataframe
busy_df = pd.read_csv(file_path, names=['date', 'assignment', 'exam'], skiprows=1)
print(busy_df.head())
busy_df['total'] = busy_df['exam'] + busy_df['assignment']
print(busy_df)

######################################################## hyp1 #################################################################
#data gathering

# Load the JSON file
with open(r'insta_data\your_instagram_activity\likes\liked_posts.json') as file:
    like_data = json.load(file)
with open(r'insta_data\your_instagram_activity\saved\saved_posts.json') as file:
    save_data = json.load(file)

# Extract the relevant information
liked_posts = []
saved_posts = []

limitToTerm = True
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

# Convert the 'date' column to datetime
df_likes['date'] = pd.to_datetime(df_likes['date'])
df_saves['date'] = pd.to_datetime(df_saves['date'])

# Group by day and count the number of likes/saves
likes_per_day = df_likes.groupby(df_likes['date'].dt.date).size().reset_index(name='likes')

saves_per_day = df_saves.groupby(df_saves['date'].dt.date).size().reset_index(name='saves')

# Merge the likes_per_week and saves_per_week DataFrames
df_total = pd.merge(likes_per_day, saves_per_day, on='date', how='outer')
df_total = df_total.fillna(0)
df_total['total'] = df_total['likes'] + df_total['saves']

####################################################### model training ##################################################################

df_total['total_category'] = pd.cut(df_total['total'], bins=[0, 7, 20, np.inf], labels=[0, 1, 2])#low, intermediate, high
df_total['exam']= busy_df['exam']
df_total['assignment']= busy_df['assignment']
df_total['weighted_total']= (busy_df['exam']*3+busy_df['assignment'])/4
df_total.set_index('date', inplace=True)
df_total = df_total.drop(columns=['likes', 'saves', 'total'])

print(df_total)

df_shuffled = df_total.sample(frac=1, random_state=42)  # Setting a random seed for reproducibility
# Separate dependent variable (y) and independent variables (X)
X = df_shuffled.drop(columns=['total_category'])
y = df_shuffled['total_category']

# Split the dataset into training and test sets (65% training, 35% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

correlation_matrix = df_total.corr()

# Highlight strong correlations with the target variable 'health_metrics'
target_correlations = correlation_matrix['total_category'].sort_values(ascending=False)

# Plot the results in a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Display strong correlations with the target variable
print("Correlations with 'total_category':")
print(target_correlations)

# Define the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20]
}

# Create GridSearchCV object
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(X, y)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

dt_classifier = DecisionTreeClassifier(max_depth=3, min_samples_split=20, random_state=42)

# Train the model with the chosen hyperparameters
dt_classifier.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['low', 'intermediate', 'high'], rounded=True, fontsize=10)
plt.title("Decision Tree with Hyperparameters")
plt.show()

# Predict labels for testing data
y_pred = dt_classifier.predict(X_test)

# Get unique labels from y_test
unique_labels = sorted(y_test.unique())

# Define the new labels
new_labels = ['low', 'intermediate', 'high']

# Report the classification accuracy
accuracy = accuracy_score(y_test, y_pred)


# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

# Plot the confusion matrix as a heatmap with new labels
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=new_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"Classification Accuracy: {accuracy:.4f}")

######################################################## hyp2 #################################################################

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


incoming_df = pd.DataFrame(incoming_reaction_rates).reset_index()
outgoing_df = pd.DataFrame(outgoing_reaction_rates).reset_index()

# Reorder the columns
incoming_df = incoming_df.reindex([True, False], axis=1)
outgoing_df = outgoing_df.reindex([True, False], axis=1)

import scipy.stats as stats

test_df=pd.DataFrame()
test_df['True']=incoming_df[True]
test_df['False']=incoming_df[False]

# Calculate the p-values and store them in a matrix
p_values = []
for i in range(6):
    row = []
    for j in range(i):
        row.append(np.NaN)
    for k in range(i+1, 7):
        oddsratio, p = stats.fisher_exact(test_df.iloc[[i,k]])
        row.append(p)
    p_values.append(row)
p_df= pd.DataFrame(p_values)
p_df.columns= p_df.columns+1
print(p_df)

# Plot the results in a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(p_df, annot=True, cmap='coolwarm', fmt='.8f')  # Increase precision to 4 decimal places
plt.title('Heatmap of p values of Fisher\'s Exact Test')
plt.show()