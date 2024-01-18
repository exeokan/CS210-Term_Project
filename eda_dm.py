#second hyp could be i prioritize some interactions more

import pandas as pd
import json
import numpy as np
from datetime import datetime
import seaborn as sns
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DM_file_names=['insta_data\your_instagram_activity\messages\inbox\ardabaristonbil_933043168094634\message_1.json',
           'insta_data\your_instagram_activity\messages\inbox\aylinozkaya_496071265125162\message_1.json',
           'insta_data\your_instagram_activity\messages\inbox\mertpolat_496349915097297\message_1.json',
           'insta_data\your_instagram_activity\messages\inbox\taha_813439610054991\message_1.json',
           'insta_data\your_instagram_activity\messages\inbox\utku_906578497407768\message_1.json']

group_file_names=['insta_data\your_instagram_activity\messages\inbox\bucalismaz_5324286234349409\message_1.json',
               'insta_data\your_instagram_activity\messages\inbox\adpafterlife_5939229612835128\message_1.json']

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

