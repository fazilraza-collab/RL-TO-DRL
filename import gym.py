import gymnasium as gym
import numpy as np 
eliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")

done = False
state = eliffEnv.reset()
while not done:
    print(eliffEnv.render())
    action = int(np.random.randint(low=0, high=4, size=1))
    print(state, "-->",action)
    state, reward, terminated, truncated, info = eliffEnv.step(action)
    done = terminated or truncated
eliffEnv.close() 