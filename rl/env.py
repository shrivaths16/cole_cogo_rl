import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import matplotlib.pyplot as plt
from gym import envs

# Create the environment
print(envs.registry.all())
env = gym.make('antmaze-umaze-v2')
obs  = env.reset()
for _ in range(50):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(reward, done, info)
plt.plot(obs)
plt.show()