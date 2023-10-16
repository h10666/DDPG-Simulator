import numpy as np
from gym import spaces
import tensorflow as tf

high = [np.finfo(np.float32).max, np.finfo(np.float32).max]
low = [0, 0]
high = np.array(high)
low = np.array(low)
observation_space = spaces.Box(low, high)
print(observation_space.shape[-1])
a = [[128, 10]]
print(np.array(a).reshape([-1, observation_space.shape[-1]]))