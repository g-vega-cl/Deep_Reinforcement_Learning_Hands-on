import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np





"""
This class is a wrapper around a Gym game, which includes several transformations:

Resize input image from 210 × 160 (standard Atari resolution) to a square size 64 × 64

Move color plane of the image from the last position to the first, 
	to meet the PyTorch convention of convolution layers that input a 
	tensor with the shape of channels, height, and width

Cast the image from bytes to float and rescale its values to a 0..1 range

"""
IMAGE_SIZE = 64
class InputWrapper(gym.ObservationWrapper):
	def __init__(self, *args):
		super(InputWrapper, self).__init__(*args)
		assert isinstance(self.observarion_space, gym.spaces.Box) #gym.spaces.box is a type of space
		old_space = self.observation_space #Observation space comes with the gym environment
		self.observarion_space = gym.spaces.Box(self.observation(old_space.low),
			self.observation(old_space.high), dtype = np.float32)

	def observation(self, observation):
		#resize image
		new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
		# transform (210, 160, 3) -> (3, 210, 160)
		return new_obs.astype(np.float32) / 255

"""
Then we define two nn.Module classes: Discriminator and Generator. 
The first takes our scaled color image as input and, 
by applying five layers of convolutions, 
converts it into a single number, 
passed through a sigmoid nonlinearity. 
The output from Sigmoid is interpreted as 
the probability that Discriminator thinks our input image is from the real dataset.
"""



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)

"""
Generator takes as input a vector of random numbers (latent vector) 
and using the "transposed convolution" operation 
(it is also known as deconvolution), 
converts this vector into a color image of the original resolution.

I think that means that it literally generates images from random numbers
"""

class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)




"""
This infinitely samples the environment from the provided array, 
issues random actions and remembers observations in the batch list. 
When the batch becomes of the required size, we convert it to a tensor 
and yield from the generator. The check for the nonzero mean of the observation 
is required due to a bug in one of the games to prevent the flickering of an image.
"""
BATCH_SIZE = 16
def iterate_batches(envs, batch_size = BATCH_SIZE):
	batch = [e.reset() for e in envs]
	env_gen = iter(lambda: random.choice(envs), None)

	while True:
		e = next(env_gen)
		obs, reward, is_done, _ = e.step(e.action_space.sample())
		if np.mean(obs) > 0.01:
			batch.append(obs)
		if len(batch) == batch_size:
			yield torch.FloatTensor(batch)
			batch.clear()
		if is_done:
			e.reset()


#THE MAIN FUNCTION

LEARNING_RATE = 0.0001
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default = False, action = "store_true")
	args = parser.parse_args()
	device = torch.device("cuda" if args.cuda else "cpu")
	env_names = ('Breakcout-v0', 'AirRaid-v0', 'Pong-v0')
	envs = [InputWrapper(gym.make(name)) for name in env_names]
	input_shape = envs[0].observation_space.shape
	#we process command-line arguments (--cuda).
	#create environment pool with wrapper applied.
	#THIS WILL BE PASSED TO ITERATE_BATCHES TO GENERATE TRAINING DATA

	Writer = SummaryWriter() #Summary writer
	net_discr = Discriminator(input_shape=input_shape).to(device) #create both networks
	net_gener = Generator(output_shape=input_shape).to(device)

	objective = nn.BCELoss() #The loss
	#Note: there are two optimizers because GANs get trained that way
	gen_optimizer = optim.Adam(params = net_gener.parameters(), lr = LEARNING_RATE) 
	dis_optimizer = optim.Adam(params = net_discr.parameters(), lr = LEARNING_RATE)
	
	#We are now defining arrays to accumulate losses, iteration count and labels
	gen_losses = []
	dis_losses = []
	iter_no = 0

	#Basically the real ones are ones, the fake ones are zeroes
	true_labels_v = torch.ones(BATCH_SIZE,dtype=torch.float32, device = device)
	fake_labels_v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)	

	#Train the model- THE TRAINING LOOP
	for batch_v in iterate_batches(envs):
		#Generate extra fake samples, input is 4D: batch, filters, x,y
			#And normalize
		gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE,1,1).normal_(0,1).to(device)
		batch_v = batch_v.to(device)
		gen_output_v = net_gener(gen_input_v)

		#Train discriminator
			#We train it by applying it two times, to the true data and to the generated ones, 
			#We need to call the detach() function on the generator's output  to
			#prevent gradients of this training pass from flowing into the generator 
			#(detach() is a method of tensor, which makes a copy of it without connection to the parent's operation).
		dis_optimizer.zero_grad()
		dis_output_true_v = net_discr(batch_v)
		dis_output_fake_v = net_discr(gen_output_v.detach()) 
		#what does the objective function does? I think we are summing the 
			#losses of how accurate were the true and false labels.
		dis_loss = objective(dis_output_true_v,true_lavels_v) + objective(dis_output_fake_v, fake_labels_v)
		diss_loss.backward()
		dis_optimizer.step()
		dis_losses.append(dis_loss.item())

