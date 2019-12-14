import torch
import torch.nn as nn

class OurModule(nn.Module):
	def __init__(self, num_inputs, num_classes, dropout_prob = 0.3):
		super(OurModule,self).__init__()
		self.pipe = nn.Sequential(
				nn.Linear(num_inputs,5), #a simple network for any number of inputs
				nn.ReLU(),
				nn.Linear(5,20),
				nn.ReLU(),
				nn.Linear(20, num_classes), #num_classes is what we want our output to be
				nn.Dropout(p = dropout_prob),
				nn.Softmax(dim = 1)
			)

	def forward(self,x):
		return self.pipe(x)

if __name__ == "__main__":
	net = OurModule(num_inputs = 2, num_classes = 3)
	v = torch.FloatTensor([[2,3]])
	out = net(v)
	print(net)
	print(out)

"""
A FEW THINGS TO MONITOR
DL practitioners have developed a list of things that you should observe during your training, which usually includes the following:

Loss value, which normally consists of several components like base loss and regularization losses. You should monitor both total loss and individual components over time.
Results of validation on training and test sets.
Statistics about gradients and weights.
Learning rates and other hyperparameters, if they are adjusted over time.
"""