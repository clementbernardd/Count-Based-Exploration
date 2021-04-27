# Count-Based-Exploration
Our version of #Exploration: A Study of Count-Based Explorationfor Deep Reinforcement Learning for a class project : https://arxiv.org/pdf/1611.04717v3.pdf

This aim of the project was to implement the count-based exploration with the static hashing with Deep RL algorithms. 

We study the differences between the epsilon-greedy exploration for 2 tabular RL algorithms (Q-learning and Sarsa) and 3 Deep RL algorithms (DQN, DDQN and Dueling DQN). 

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

The histograms of the states are quite balanced and the rewards show that both methods of exploration work well. 


## Acrobot 

### Rewards

Here are the rewards for DQN, DDQN and Double DQN for the Acrobot environment. 

| DQN | DDQN | Dueling DQN |  
|---| --- | --- | 
| ![](/images/acrobot_dqn.png) | ![](/images/acrobot_ddqn.png) | ![](/images/acrobot_dueling_dqn.png) |

All the methods work well, and it seems that the count-based methods is slightly faster than the epsilon-greedy exploration. 

### Histograms 

Here are the histograms of the states for DQN, DDQN and Dueling DQN 

| DQN | DDQN | Dueling DQN |   
|---| --- | --- | 
| ![](/images/acrobot_state_dqn.png) | ![](/images/acrobot_state_ddqn.png) | ![](/images/acrobot_state_dueling_dqn.png) |

We see that few states are more visited than others, but the behaviour seems correct and the model has learnt well. 

### Simulation 

Here is the simulation with the model of SEED 77 on 10 instances of the environment. 


| Random |DQN (count-based) | DDQN (count-based) | Dueling DQN (count-based) |   
|--- | ---| --- | --- | 
|![](/gif/acrobot/random/random.gif)|![](/gif/acrobot/dqn/count_based.gif) | ![](/gif/acrobot/ddqn/count_based.gif) | ![](/gif/acrobot/dueling_dqn/count_based.gif) |
|--- | ---| --- | --- | 
| Random |DQN (epsilon-greedy) | DDQN (epsilon-greedy) | Dueling DQN (epsilon-greedy) |    
|![](/gif/acrobot/random/random.gif)|![](/gif/acrobot/dqn/epsilon_greedy.gif) | ![](/gif/acrobot/ddqn/epsilon_greedy.gif) | ![](/gif/acrobot/dueling_dqn/epsilon_greedy.gif) |

They all perform well except the count-based DQN. It should be explain by the fact we used the model trained on SEED 77, and could have had struggle to learn because of bad initialisation. 

#### Average rewards

Here are the average rewards on 10 instance of the environment 

| Mean rewards | 
| --- | 
| ![](images/mean_acrobot.png)|



## Cartpole 

### Rewards

Here are the rewards for DQN, DDQN and Double DQN for the Cartpole environment. 

| DQN | DDQN | Dueling DQN |  
|---| --- | --- | 
| ![](/images/cartpole_dqn.png) | ![](/images/cartpole_ddqn.png) | ![](/images/cartpole_dueling_dqn.png) |

The count-based method doesn't work well for this environment. It could be explain by a bad static hashing due to the fact that there aren't a lot of dimensions on the original continuous state space. 


### Histograms 

Here are the histograms of the states for DQN, DDQN and Dueling DQN

| DQN | DDQN | Dueling DQN |   
|---| --- | --- | 
| ![](/images/cartpole_state_dqn.png) | ![](/images/cartpole_state_ddqn.png) | ![](/images/cartpole_state_dueling_dqn.png) |

As there is logarithm scale, we should consider that some states are highly more visited than others. It could explain than too much states are discretize into same representation, which unable the model to explore well.  

### Simulation 

Here is the simulation with the model of SEED 77 on 10 instances of the environment. 


| Random |DQN (count-based) | DDQN (count-based) | Dueling DQN (count-based) |   
|--- | ---| --- | --- | 
|![](/gif/cartpole/random/random.gif)|![](/gif/cartpole/dqn/count_based.gif) | ![](/gif/cartpole/ddqn/count_based.gif) | ![](/gif/cartpole/dueling_dqn/count_based.gif) |
|--- | ---| --- | --- | 
| Random |DQN (epsilon-greedy) | DDQN (epsilon-greedy) | Dueling DQN (epsilon-greedy) |    
|![](/gif/cartpole/random/random.gif)|![](/gif/cartpole/dqn/epsilon_greedy.gif) | ![](/gif/cartpole/ddqn/epsilon_greedy.gif) | ![](/gif/cartpole/dueling_dqn/epsilon_greedy.gif) |

Only the epsilon-greedy version performs well. The discretisation with the static hashing isn't sufficiently granular. 

#### Average rewards

Here are the average rewards on 10 instance of the environment 

| Mean rewards | 
| --- | 
| ![](images/mean_cartpole.png)|

The count-based method isn't working for this environment. 


## Mountain car 

### Rewards

Here are the rewards for DQN, DDQN and Double DQN for the Mountain car environment. 

| DQN | DDQN | Dueling DQN |  
|---| --- | --- | 
| ![](/images/mountain_dqn.png) | ![](/images/mountain_ddqn.png) | ![](/images/mountain_dueling_dqn.png) |

The count-based method outperforms the epsilon-greedy exploration for both DQN and DDQN, but not for Dueling DQN. Such a behaviour should be explain by some struggles in the choise of the count-based hyperparameters. 


### Histograms 

Here are the histograms of the states for DQN, DDQN and Dueling DQN

| DQN | DDQN | Dueling DQN |   
|---| --- | --- | 
| ![](/images/mountain_state_dqn.png) | ![](/images/mountain_state_ddqn.png) | ![](/images/mountain_state_dueling_dqn.png) |

We see that there are around 30 different discrete states for DQN and DDQN, and 4 times more for Dueling DQN. Maybe the bad behaviour of the Dueling DQN could be explain by the sparsity of the static hashing. 

### Simulation 

Here is the simulation with the model of SEED 77 on 10 instances of the environment. 


| Random |DQN (count-based) | DDQN (count-based) | Dueling DQN (count-based) |   
|--- | ---| --- | --- | 
|![](/gif/mountain/random/random.gif)|![](/gif/mountain/dqn/count_based.gif) | ![](/gif/mountain/ddqn/count_based.gif) | ![](/gif/mountain/dueling_dqn/count_based.gif) |
|--- | ---| --- | --- | 
| Random |DQN (epsilon-greedy) | DDQN (epsilon-greedy) | Dueling DQN (epsilon-greedy) |    
|![](/gif/mountain/random/random.gif)|![](/gif/mountain/dqn/epsilon_greedy.gif) | ![](/gif/mountain/ddqn/epsilon_greedy.gif) | ![](/gif/mountain/dueling_dqn/epsilon_greedy.gif) |

Only the Dueling DQN with count-based exploration doesn't solve the environment. 

#### Average rewards

Here are the average rewards on 10 instance of the environment 

| Mean rewards | 
| --- | 
| ![](images/mean_mountain.png)|





































































# Description 

- `gif` : The gif folder where all the results of the models are saved
- `images` : The images used for either the notebooks or the report
- `models` : The models parameters 
- `notebooks` : The notebooks used during the project
- `python` : The python files used (like the architectures of the models)
- `requirements.txt` : The version of the library used
- `results` : The results of the training 
- `zip` : A zip file where the `gif`, `models`, `python` and `results` are compressed to be used in a notebook










