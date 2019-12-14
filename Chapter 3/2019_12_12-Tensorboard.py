import math
from tensorboardX import SummaryWriter

if __name__ == "__main__":
	writer = SummaryWriter()

	funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

	for angle in range(-360, 360):
		angle_rad = angle * math.pi / 180
		for name, fun in funcs.items():
			val = fun(angle_rad)
			writer.add_scalar(name, val, angle)

	writer.close()

"""
Here, we loop over angle ranges in degrees, 
convert them into radians, and calculate our functions' values. 
Every value is being added to the writer using the add_scalar function, 
which takes three arguments: the name of the parameter, 
its value, and the current iteration (which has to be an integer).
"""

"""
The result of running this will be zero output on the console, 
but you will see a new directory created inside the runs directory with a single file. 
To look at the result, we need to start TensorBoard:

rl_book_samples/Chapter03$ tensorboard --logdir runs --host localhost
TensorBoard 0.1.7 at http://localhost:6006 (Press CTRL+C to quit)
Now you can open http://localhost:6006 in your browser to see something like this:

as usual, needs tweeking to work.
python -m tensorboard.main --logdir="./runs"
"""