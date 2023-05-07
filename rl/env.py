import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import matplotlib.pyplot as plt
from gym import envs

# Create the environment
print(envs.registry.all())
env = gym.make('antmaze-medium-diverse-v2', expose_body_coms=True)
obs = env.reset()
plt.plot(obs)
plt.show()