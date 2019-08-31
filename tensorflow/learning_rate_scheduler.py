import numpy as np
from keras.callbacks import LearningRateScheduler
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

## Learning rate decay whhich needs to be called from callback
### How to call : 	schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=15)
### callbacks = [LearningRateScheduler(schedule)]

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule)


# Each and every learning rate schedule we implement will have a plot function, enabling us to visualize our learning rate over time.
# With our base LearningRateSchedule  class implement, let’s move on to creating a step-based learning rate schedule.
### How to call : schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)
### callbacks = [LearningRateScheduler(schedule)]
class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]

		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")

# Where \alpha_{I} is the initial learning rate, F is the factor value controlling the rate in which the learning date drops, D is the “Drop every” epochs value,
# and E is the current epoch.

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery

	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)

		# return the learning rate
		return float(alpha)

## Linear and polynomial learning rate schedules in Keras
## Using these methods our learning rate is decayed to zero over a fixed number of epochs.
##The rate in which the learning rate is decayed is based on the parameters to the polynomial function.
## # A smaller exponent/power to the polynomial will cause the learning rate to decay “more slowly”, whereas larger exponents decay the learning rate “more quickly”.

# maxEpochs : The total number of epochs we’ll be training for.
# initAlpha : The initial learning rate.
# power : The power/exponent of the polynomial.
## Note that if you set power=1.0  then you have a linear learning rate decay.

### How to call : 	schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)  -- This is pollynomial
###  How to call : 	PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1) -- This is linear
### callbacks = [LearningRateScheduler(schedule)]

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power

	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay

		# return the new learning rate
		return float(alpha)



