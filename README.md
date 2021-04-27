# Count-Based-Exploration
Our version of #Exploration: A Study of Count-Based Explorationfor Deep Reinforcement Learning for a class project : https://arxiv.org/pdf/1611.04717v3.pdf

This aim of the project was to implement the count-based exploration with the static hashing with Deep RL algorithms. 

We study the differences between the epsilon-greedy exploration for 2 tabular RL algorithms (Q-learning and Sarsa) and 3 Deep RL algorithms (DQN, DDQN and Dueling-DQN). 

Here are some results of our work : 

| Acrobot | Cartpole | Mountain | 
|---| --- | --- |
| ![](/gif/acrobot/dueling_dqn/count_based.gif) | ![](/gif/cartpole/ddqn/epsilon_greedy.gif) | ![](/gif/mountain/ddqn/count_based.gif)   |
| Dueling DQN with count based exploration | DDQN with epsilon-greedy exploration  | DDQN with count based method | 


# Count-based 

The count-based exploration uses a static hashing to map continuous states into discrete state, and then count the number of times a given state has been visited. 
Then, the classic RL algorithms are trained with a bonus reward that takes into account the number of times we have visited the state. This bonus reward plays the role of the exploration. 



## Static hashing

The static hashing maps the continuous states into discrete with a A matrix which is drawn from normal distribution as explained in the following figure. 


| Static hashing |  
|---| 
| ![](/images/static_hashing_logo.png) | 




## Counting 

Once we have discretize the continuous states, we use a dictionnary to count the number of times we have visited a given state. There are two hyperparameters that are used in the algorithm : 
- `beta` : Tells how important we consider the bonus reward
- `k` : The granularity of the static hashing. It should be sparse enough to keep distant non close states but not too much to merge the close states in the continuous space. 


| Count-based |  
|---| 
| ![](/images/count_based_logo.png) | 


# Results 

We have trained 2 tabular methods (Q-learning and Sarsa) and 3 Deep RL methods (DQN, DDQN and Dueling DQN) on 4 different environments : Taxi, Acrobot, Cartpole and Mountain Car. 

We have trained each model on the environments with 10 SEEDS and stored the rewards. After that, we plot the average rewards and the variance of the rewards over the episodes. 

We have trained our models with the same hyperparameters for both epsilon-greedy and count based explorations. The hyperparameters are summarised in the `Summary.ipynb` notebook. 


## Taxi 

#### Rewards

Here are the rewards for Q-learning and Sarsa. 

| Q-learning | Sarsa |  
|---| --- | 
| ![](/images/taxi_q_learning.png) | ![](/images/taxi_sarsa.png) | 
 
### Histograms 

Here are the histograms of the states for both Q-learning and Sarsa for the count-based method 

| Q-learning | Sarsa |  
|---| --- | 
| ![](/images/taxi_state_q_learning.png) | ![](/images/taxi_state_sarsa.png) | 


























# Description 

- `gif` : The gif folder where all the results of the models are saved
- `images` : The images used for either the notebooks or the report
- `models` : The models parameters 
- `notebooks` : The notebooks used during the project
- `python` : The python files used (like the architectures of the models)
- `requirements.txt` : The version of the library used
- `results` : The results of the training 
- `zip` : A zip file where the `gif`, `models`, `python` and `results` are compressed to be used in a notebook










