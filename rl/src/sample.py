import sys
from env.wrappers import make_env
env = make_env("robot", "reach")
obs = env.reset()
env.render()