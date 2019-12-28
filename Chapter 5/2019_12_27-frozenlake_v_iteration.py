import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0" #environment
GAMMA = 0.9 #gamma (decay)
TEST_EPISODES = 20 #episodes to test

"""
The overall logic of our code is simple: in the loop, we play 100 random steps from the environment, 
populating the reward and transition tables. After those 100 steps, we perform a value iteration loop over all states,
updating our value table. Then we play several full episodes to check our improvements using the updated value table. 
If the average reward for those test episodes is above the 0.8 boundary, then we stop training.
During test episodes, we also update our reward and transition tables to use all data from the environment.
"""


#Then we define the Agent class, which will keep our tables and contain functions we'll be using in the training loop:
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME) #get the environment
        self.state = self.env.reset() #reset
        self.rewards = collections.defaultdict(float) #get rewards?
        self.transits = collections.defaultdict(collections.Counter) #defaultdict returns a dic with the instance and, if repeated, their respective repetition values
        self.values = collections.defaultdict(float) #https://docs.python.org/2/library/collections.html

    """
    This function is used to gather random experience from the environment and update reward and transition tables. 
    Note that we don't need to wait for the end of the episode to start learning; we just perform N steps and remember their outcomes. 
    This is one of the differences between Value iteration and Cross-entropy, which can learn only on full episodes.
    """
    #In the class constructor, we create the environment we'll be using for data samples, obtain our first observation, and define tables for rewards, transitions, and values.
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample() #I think this returns a single possible actions
            new_state, reward, is_done, _ = self.env.step(action) #get the values of the environment after the action
            self.rewards[(self.state, action, new_state)] = reward #get the values into self.
            self.transits[(self.state, action)][new_state] += 1 #I think this is like a counter, how many states do we have
            self.state = self.env.reset() if is_done else new_state #if its not done (game not finished), get to a new state.

    def calc_action_value(self, state, action): # calculates the value of the action from the state, We will use it for two purposes: to select the best action to perform from the state and to calculate the new value of the state on value iteration.
        target_counts = self.transits[(state, action)] #Probly get the value of transits
        total = sum(target_counts.values()) #get the total counts?
        action_value = 0.0 #reset action value
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)] #fetch rewards
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state]) #The bellman equation. the average of previous and the new ones (and future ones).
        return action_value #return the value of each this action.
    """
    The next function uses the function we just described to make a decision about the best action to take from the given state. 
    It iterates over all possible actions in the environment and calculates value for every action. 
    The action with the largest value wins and is returned as the action to take. 
    This action selection process is deterministic, as the play_n_random_steps() function introduces enough exploration. 
    So, our agent will behave greedily in regard to our value approximation.
    """
    def select_action(self, state):
        best_action, best_value = None, None #start variable
        for action in range(self.env.action_space.n): #iterate through every action
            action_value = self.calc_action_value(state, action) #get the action value
            if best_value is None or best_value < action_value: #check if the current value is higher than the highest
                best_value = action_value #set global best as best
                best_action = action
        return best_action #return best action

    """
    The play_episode function uses select_action to find the best action to take and plays one full episode using the provided environment. 
    This function is used to play test episodes, during which we don't want to mess up with the current state of the main environment used to gather random data. 
    """
    def play_episode(self, env): #just plays an episode with our set actions?
        total_reward = 0.0 #set reward of episode to 0
        state = env.reset() #reset environment
        while True: #Episode loop
            action = self.select_action(state) #get the action with the  highest value from the state
            new_state, reward, is_done, _ = env.step(action) #get reward and next state
            self.rewards[(state, action, new_state)] = reward #set the class with the new values
            self.transits[(state, action)][new_state] += 1 #add value to transits
            total_reward += reward #add new reward
            if is_done: #check if game is over
                break
            state = new_state #set new state as state
        return total_reward #return the reward of the episode
    """
    What we do is just loop over all states in the environment, 
    then for every state we calculate the values for the states reachable from it,
    obtaining candidates for the value of the state. 
    Then we update the value of our current state with the maximum value of the action available from the state:
    """
    #I think this just merges the prev functions and decides the best action
    def value_iteration(self): 
        for state in range(self.env.observation_space.n): #iterate for every space in the observation
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)] #calculate the value of every action
            self.values[state] = max(state_values) #get the max of the possible values

#That's all our agent's methods, and the final piece is a training loop and the monitoring of the code:
if __name__ == "__main__": #Note, this is outside the class
    test_env = gym.make(ENV_NAME) #get env
    agent = Agent() #start agent (our class)
    writer = SummaryWriter(comment="-v-iteration") #start writer

    iter_no = 0 #iteration 0
    best_reward = 0.0 #the best reward
    while True: #start loop
        iter_no += 1 #add iteration
        agent.play_n_random_steps(100) #First, we perform 100 random steps to fill our reward and transition tables with fresh data 
        agent.value_iteration() #and then we run value iteration over all states.

        reward = 0.0 #restart reward
        for _ in range(TEST_EPISODES): #for the number of test episodes
            reward += agent.play_episode(test_env) #play the episode of the environment and add the reward
        reward /= TEST_EPISODES #divide the reward over the test episodes
        writer.add_scalar("reward", reward, iter_no) #add reward to the writer
        if reward > best_reward: #if we get a higher reward rewrite and print
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80: #if we get a higher reward than 0.8 its over, we win.
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()