import gymnasium as gym
import tensorflow as tf
from keras.models import load_model
import time

# 1. Load the trained model
# IMPORTANT: Change this filename to match exactly what you saved it as!
# (e.g., "cartpole_model.keras" or "sarsa_q_net.keras")
MODEL_PATH = "sarsa_q_net.keras" 
print(f"Loading model from {MODEL_PATH}...")
q_net = load_model(MODEL_PATH)

# 2. Create the environment with human rendering
# render_mode="human" will open a graphical window showing the cart and pole
env = gym.make("CartPole-v1", render_mode="human")

# 3. Define the greedy policy (NO exploration for evaluation)
def get_action(state):
    # Convert the 4-value state into a tensor with a batch dimension: shape (1, 4)
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    
    # Get Q-values from the network (training=False ensures no dropout/batch norm issues)
    q_values = q_net(state_tensor, training=False)[0]
    
    # Choose the action with the highest Q-value (0 = Left, 1 = Right)
    action = int(tf.argmax(q_values).numpy())
    return action

# 4. Evaluation Parameters
NUM_EPISODES = 10

print(f"\nStarting evaluation for {NUM_EPISODES} episodes...")
print("Watch the window to see the agent balance the pole!\n")

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        # Get the best action from our trained model
        action = get_action(state)
        
        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update state and stats
        state = next_state
        total_reward += reward
        episode_length += 1
        
        # Optional: Uncomment the line below if the animation runs too fast to see
        # time.sleep(0.02) 

    print(f"Episode: {episode + 1:2d} | Length: {episode_length:3d} | Total Reward: {total_reward:3.0f}")

print("\nEvaluation complete!")
env.close()