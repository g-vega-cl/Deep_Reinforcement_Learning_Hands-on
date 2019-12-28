#!/usr/bin/env python3
"""
he difference is really minor. 
The most obvious change is to our value table. 
In the previous example, we kept the value of the state, so the key in the dictionary was just a state. 
Now we need to store values of the Q-function, which has two parameters: state and action, so the key in the value table is now a composite.
"""
"""
The second difference is in our calc_action_value function.
 We just don't need it anymore, as our action values are stored in the value table. 
 Finally, the most important change in the code is in the agent's value_iteration method. 
 Before, it was just a wrapper around the calc_action_value call, which did the job of Bellman approximation. 
 Now, as this function has gone and was replaced by a value table, we need to do this approximation in the value_iteration method.

"""

import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
    """
    The code is very similar to calc_action_value in the previous example and in fact it does almost the same thing. 
    For the given state and action, it needs to calculate the value of this action using statistics about target states that we've reached with the action. 
    To calculate this value, we use the Bellman equation and our counters, which allow us to approximate the probability of the target state. 
    However, in Bellman's equation we have the value of the state and now we need to calculate it differently. 
    Before, we had it stored in the value table (as we approximated the value of states), so we just took it from this table. 
    We can't do this anymore, so we have to call the select_action method, which will choose for us the action with the largest Q-value, and then we take this Q-value as the value of the target state. 
    Of course, we can implement another function which could calculate for us this value of state, but select_action does almost everything we need, so we will reuse it here.
    """
    def value_iteration(self): #This is the main difference between this one and the _v_ iteration
        for state in range(self.env.observation_space.n): #iterate for every space in the observation
            for action in range(self.env.action_space.n): #for every action do:
                action_value = 0.0 #reset action value
                target_counts = self.transits[(state, action)] #get the target_counts? what are those?
                total = sum(target_counts.values()) #get the sum of all target_counts
                for tgt_state, count in target_counts.items(): #for each value of the items in target counts:
                    reward = self.rewards[(state, action, tgt_state)] #get the reward 
                    best_action = self.select_action(tgt_state) #get the best action using the select_action function (same as prev code)
                    action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)]) #Bellman to calculate prev action
                self.values[(state, action)] = action_value #set the class value as the action value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
