import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128 #Hidden neurons
BATCH_SIZE = 16 #Count of episodes in every iteration
PERCENTILE = 70 #Percentile of opisodes total rewards (what we keep)


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), #(in_features, out_features)
            nn.ReLU(), #https://pytorch.org/docs/stable/nn.html
            nn.Linear(hidden_size, n_actions) #get the output, which are your actions.
        )

    def forward(self, x): #This just passes the net, it seems, but also, i dont see us using it anywhere.
        return self.net(x)

#This is a single episode stored as total undiscounted reward and a collection of EpisodeStep.
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
#This will be used to represent one single step that our agent made in the episode, and it stores the observation from the environment and what action the agent completed. We'll use episode steps from elite episodes as training data.
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size): #We create the batches it seems
    batch = [] #create batch
    episode_reward = 0.0 #start a new reward
    episode_steps = [] #start a new steps array
    obs = env.reset() #Create a new environment
    sm = nn.Softmax(dim=1) #This is the last function it converts the network's output to a probability distribution of actions. 
    while True: #The environment loop
        obs_v = torch.FloatTensor([obs]) #Get the observation before action
        act_probs_v = sm(net(obs_v)) #pass the neural net result to a Softmax that returns one dimentional action probability
        act_probs = act_probs_v.data.numpy()[0] #Transform the data into a readable array by numpy?
        action = np.random.choice(len(act_probs), p=act_probs) #With the probabilities get a weighted 'random' choice
        next_obs, reward, is_done, _ = env.step(action) #Ask the environment for the next state, the reward and if its over.
        episode_reward += reward #add the reward to the current episode (episode is one play before game over)
        episode_steps.append(EpisodeStep(observation=obs, action=action)) # it adds to the global array the observation and the action, to 'remember' what it has done?
        if is_done: #when the game finishes (either loss or win)
            batch.append(Episode(reward=episode_reward, steps=episode_steps)) #It appends an array of all actions performed (with their respective observations) and the total reward 
            episode_reward = 0.0 #Reset episode reward
            episode_steps = [] #Reset step array
            next_obs = env.reset() #Reset environment
            if len(batch) == batch_size: #If we reach the batch size
                yield batch #yield is like return but it continues where it left: https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/   https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
                    #(it's handy when you know your function will return a huge set of values that you will only need to read once.)
                batch = [] #reset batch
        obs = next_obs #get next obs to start the loop again in the next frame


def filter_batch(batch, percentile):  #
    #I think this is making the rewards a list (from the batch), map is appliying the function s: s.reward to every batch, which is transforming every observation into its reward
    rewards = list(map(lambda s: s.reward, batch)) #Not sure https://www.w3schools.com/python/python_lambda.asp  lambda seems to just be a function but fast
    reward_bound = np.percentile(rewards, percentile) #Returns the q-th percentile(s) of the array elements. https://docs.scipy.org/doc/numpy/reference/generated/numpy.percentile.html
    reward_mean = float(np.mean(rewards)) #Get the average of the rewards

    train_obs = [] #Create array of training observations
    train_act = [] #Create array of training actions
    for example in batch: #For each episode in batch
        if example.reward < reward_bound: #Only if the reward is better (here lower is better?) we continue
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps)) #adds every observation from the example (current episode in batch) to the train_obs 
        train_act.extend(map(lambda step: step.action, example.steps)) #adds every action from the example (current episode in batch) to the train_act

    train_obs_v = torch.FloatTensor(train_obs) #A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
    train_act_v = torch.LongTensor(train_act) #The float and long indicate the type of data. I guess its just so we can pass to the NNet easily.
    return train_obs_v, train_act_v, reward_bound, reward_mean 


if __name__ == "__main__": #Main function, where all our functions come togheter
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0] #Take observations from the environment.
    n_actions = env.action_space.n #take the possible actions from the environemnt.

    net = Net(obs_size, HIDDEN_SIZE, n_actions) #Take the parameters from the environment and our own parameter and build the neural network
    objective = nn.CrossEntropyLoss() #Set the objective .... https://pytorch.org/docs/stable/nn.html ... It is useful when training a classification problem with C classes. If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set.
    optimizer = optim.Adam(params=net.parameters(), lr=0.01) #Using adam optimmizer
    writer = SummaryWriter(comment="-cartpole") #This is for tensorboard and checking progress... command is: python -m tensorboard.main --logdir="./runs"

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)): #For each iteration and batch it enumerates them??. 
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE) #It removes the 'less fit episodes' from the batch
        optimizer.zero_grad() # Sets gradients of all model parameters to zero. https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        action_scores_v = net(obs_v) #get the observations after the action?
        loss_v = objective(action_scores_v, acts_v) #from the crossEntropyLoss get the loss. taken from your actions and the scores they got.
        loss_v.backward() #This is for backpropagation?? thats why we need zero_grad??
        optimizer.step() #CrossEntropyLoss take step?
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (  #Basically write whats happening
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199: #Once we get to this reward, we win.
            print("Solved!")
            break
    writer.close()