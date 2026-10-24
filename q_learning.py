import gymnasium as gym
import numpy as np
import pickle as pk

eliffEnv = gym.make("CliffWalking-v1")

q_table = np.zeros(shape=(48,4))



#Parameters

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODE = 500


def policy(state ,explore = 0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = np.random.randint(0, 4)

    return action 


for episode in range (NUM_EPISODE):
    done = False 
    total_reward = 0 
    episode_length = 0
    state, info = eliffEnv.reset()
    


    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, info = eliffEnv.step(action)
        done = terminated or truncated

        next_action = policy(next_state, )



        q_table[state][action] += ALPHA *(reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
        
        state = next_state
        action = next_action
        
        total_reward += reward
        episode_length +=1
        
    print("Episode:", episode,"Episode Length:", episode_length,"Total_Reward:", total_reward)
        
eliffEnv.close()
        
pk.dump(q_table, open("q_learning_q_table.pk","wb"))
print ("Training Complete. Q table Saved")