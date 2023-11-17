##Line to be added before running the script:
#  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 (with UBUNTU 22.04)
import random
import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# env = gym.make("CartPole-v1", render_mode = "human")
env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n


print(f"No. of States : {states}")
print(f"No. of Actions : {actions}")

model = Sequential()
model.add(Flatten(input_shape = (1, states)))
model.add(Dense(26, activation = "relu"))
model.add(Dense(26, activation = "relu"))
model.add(Dense(actions, activation = "linear"))

agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit=50000, window_length = 1),
    policy = BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics =["mae"])
agent.fit(env, nb_steps= 10000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize = True)

print(np.mean(results.history["episode_reward"]))

env.close();

##Initial Setup
# episodes = 10
# for ep in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         action = random.choice([0, 1])
#         print(f"\tAction Taken:{action}")
#         _, reward, done, _ = env.step(action)
#         score += reward
#         env.render()
#     print(f"Episode No.{ep}, Score: {score}")
    