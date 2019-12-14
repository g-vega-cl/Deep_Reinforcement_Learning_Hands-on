import random

class Environment:
	def __init__(self):
		self.steps_left = 10

	def get_observation(self):
		return [0,0,0] #Its 0 because we basically have no environment

	def get_actions(self):
		return [0,1]

	def is_done(self):
		return self.steps_left == 0

	def action(self,action): #Handles action and returns reward
		if self.is_done():
			raise Exception("Game is over")
		self.steps_left -= 1
		return random.random()


class Agent:
	def __init__(self):
		self.total_reward = 0.0

	def step(self,env):
		current_obs = env.get_observation() #observe
		actions = env.get_actions() #make desition
		reward = env.action(random.choice(actions)) #submit action
		self.total_reward += reward #get reward



if __name__ == "__main__": #glue code, creates both classes and runs one episode
	env = Environment()
	agent = Agent()

	while not env.is_done():
		agent.step(env)

	print("Total reward got: %.4f" % agent.total_reward)

"""
but the basic pattern stays the same: 
on every step, an agent takes some observations 
from the environment, does its calculations,
and selects the action to issue.
The result of this action is a reward and new observation.

"""