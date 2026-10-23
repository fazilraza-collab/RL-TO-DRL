import gymnasium as gym
import numpy as np
import pickle as pk
import cv2

eliffEnv = gym.make("CliffWalking-v1")

q_table = pk.load(open("sarsa_q_table.pk","rb"))


CELL_SIZE = 60
ROWS, COLS = 4, 12

def initialize_frame():
    frame = np.ones(
        (ROWS * CELL_SIZE, COLS * CELL_SIZE, 3),
        dtype=np.uint8
    ) * 255

    for row in range(ROWS):
        for col in range(COLS):
            y1, x1 = row * CELL_SIZE, col * CELL_SIZE
            y2, x2 = y1 + CELL_SIZE, x1 + CELL_SIZE

            if row == 3 and 1 <= col <= 10:
                frame[y1:y2, x1:x2] = (0, 0, 255)
            elif (row, col) == (3, 11):
                frame[y1:y2, x1:x2] = (0, 255, 0)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 0, 0), 1
            )

    return frame


def put_agent(img, state):
    row, col = divmod(int(state), COLS)

    center_x = col * CELL_SIZE + CELL_SIZE // 2
    center_y = row * CELL_SIZE + CELL_SIZE // 2

    cv2.circle(img, (center_x, center_y), 15, (255, 0, 0), -1)

    return img


def policy(state ,explore = 0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = np.random.randint(0, 4)

    return action 


#PArameters

NUM_EPISODE = 5
for episode in range(NUM_EPISODE):

    done = False
    total_reward = 0
    episode_length = 0
    frame = initialize_frame()
    state, info = eliffEnv.reset()

    while not done :
        frame2 = put_agent(frame.copy(),state)
        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)
        action = policy(state)
        next_state, reward, terminated, truncated, info = eliffEnv.step(action)

        done = terminated or truncated
        state = next_state

        episode_length += 1
        total_reward += reward
    print("Episode:", episode,"Episode Length:", episode_length,"Total_Reward:", total_reward)
    

eliffEnv.close()
cv2.destroyAllWindows()