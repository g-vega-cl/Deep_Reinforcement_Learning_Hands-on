import gym
import random 

#This is some king of general code, not specific to a single environment
class RandomActionWrapper(gym.ActionWrapper):
	def __init__(self, env, epsilon = 0.1): #The epsilon is the probability of a random action
		super(RandomActionWrapper, self).__init__(env)
		self.epsilon = epsilon #I think we are changing properties of our environment.
	
	def action(self, action):
		if random.random() < self.epsilon: #if a random number is lower than epsilon, take random action
			print("RANDOM!")
			return self.env.action_space.sample()

		return action

#This is now the cartpole:

if __name__ == "__main__":
	env = RandomActionWrapper(gym.make("CartPole-v0")) #create env

	"""
	Now it's time to apply our wrapper. 
	We will create a normal CartPole environment and pass it to our wrapper constructor.
	From here on, we use our wrapper as a normal Env instance, 
	instead of the original CartPole. 
	As the Wrapper class inherits the Env class and exposes the same interface, 
	we can nest our wrappers in any combination we want.
	"""

	obs = env.reset()
	total_reward = 0.0

	while True:
		obs, reward, done, _ = env.step(0)
		total_reward += reward
		if done:
			break

		print("Reward got: %.2f" % total_reward)