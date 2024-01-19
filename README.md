# Introduction

In this project, I am analzying my own Instagram data to learn how much I use the app through time, how do I interact with the content, and how do I do interact with my close circle. 

# Data Source

My personal Instagram data have been downloaded upon the request I made through the app itself. Downloaded package included JSON files containing various information. Instructions are available on the Instagram's website https://help.instagram.com/181231772500920.

I also prepared another dataset: number of assignments and exams I had per week in this semester (2023 Fall).

In order to relate it to the data I prepared, Instagram data is limited to after September 2023.

# Explaratory Data Analysis
After downloading, I extracted the like and save data into a pandas DataFrame in Python

1. I begun by analyzing my like statistics, and listed top 10 most liked accounts:

![mostliked](https://github.com/exeokan/CS210-Term_Project/assets/35339130/aa39211b-764d-4f21-886a-d2f5ab4d79e7)

From the graph it can be concluded that there is not much consistency to which accounts I like and they include great variance.

2. Next, I grouped number of likes and saves in a day and a week, and plotted them vs time:

![like-save per week](https://github.com/exeokan/CS210-Term_Project/assets/35339130/77893ee6-d2a6-49d0-a156-c7a3674f252f)

Firstly, i noticed there were so little saves when compared to likes, and they highly correlate with each other. I decided to take their total and only work on it when it in the hypothesis testing part.

Secondly, distribution wasn't uniform, when the school year hasn't started yet, I interacted with content more. Also, in December i had fever exams which might be the reason of the spike in that time

3. I was curious about the rate of reactions I give to the post sent to me via Direct Message, and how much reaction I recieve from others. From my inbox, I selected five person to person chats with the people from my close circle, and two group chats. These are the bar graphs of amount of posts I gave reactions to or not.
   
 ![bar_inc](https://github.com/exeokan/CS210-Term_Project/assets/35339130/b54c1e01-81d9-45f2-8772-3c946a65b743)
 
 ![bar_inc_group](https://github.com/exeokan/CS210-Term_Project/assets/35339130/e059d8f2-d7e9-4d90-b1a5-8d46946ee08a)
   
It could be seen that I prioritized giving reactions to personal DMs instead of group chats.

4. These bar graphs are the reverse version of the latter, amount of posts sent by me that got a reaction or not.
   
  ![bar_out](https://github.com/exeokan/CS210-Term_Project/assets/35339130/cba1c4d6-06f9-4803-bb3a-1d419cd44143)
  
  ![bar_out_group](https://github.com/exeokan/CS210-Term_Project/assets/35339130/e058454f-9da6-45df-b94b-787fbcbf0eb9)

It seems that chance of getting a reaction for me is higher in the group chats, since there is more people to do so.

5. Finally I plotted graphs to see if my rate of giving reaction to posts is changing over time. I plotted three graphs, seperating personal DMs and group chats, and total of personal DMs
   
  ![reaction_people](https://github.com/exeokan/CS210-Term_Project/assets/35339130/6f959a19-729f-4c04-a95a-7cf580ffb4fc)

  ![reaction_people_total](https://github.com/exeokan/CS210-Term_Project/assets/35339130/5f845d37-0dd3-46ea-a18d-fa31a9f99cbb)

  ![reaction_group](https://github.com/exeokan/CS210-Term_Project/assets/35339130/2d56ebf1-1646-444f-8655-771a654a2537)

# Hypothesis Testing

1. My first hypothesis was that my interaction total (likes+saves /day) can be predicted by how occupied I am by assignments and exams throughout the semester. For testing, I trained a decision tree using python. Here are the graphs of the correlation matrix, decision tree, and the confusion matrix.

   ![correlation_m](https://github.com/exeokan/CS210-Term_Project/assets/35339130/7ba59e03-5872-475a-9c92-7f3d1d9fb26c)
   
   ![dec_tree](https://github.com/exeokan/CS210-Term_Project/assets/35339130/f08f91a8-09ea-4e21-aeb8-596b8c4de25f)
   
   ![confusion_matrix](https://github.com/exeokan/CS210-Term_Project/assets/35339130/b6391c8a-3e6a-42f2-b810-b34ce5ee8342)

Models classification accuracy was measured as: 0.5745, which partially validates the hypothesis. I infer that since there is few weeks with much exam and assignment, the model fails to capture low and intermediate interactions 
  
2. Second hypothesis was that I prioritized giving reactions to certain chats over the other. For testing I used Fisher's Exact Test, and measured p-values for each chat pair 

![fishers](https://github.com/exeokan/CS210-Term_Project/assets/35339130/c2deed61-e91b-4d2a-be7c-0b1762327154)

For a significance level of p=0.05, it can be argued that hypothesis is valid for many of the chats. Especially my low reaction rate to group chats is evident in the graph, their p-values with the personal DMs are significant
   
