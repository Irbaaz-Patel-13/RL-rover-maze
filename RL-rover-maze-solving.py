import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Define the environment size (5x5 grid)
grid_size = 5
actions = ['up', 'down', 'left', 'right']
n_actions = len(actions)

# Define rewards
goal = (4, 4)
obstacles = [(1, 1), (2, 3), (3, 2)]

# Q-Learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
n_episodes = 1000  # Number of training episodes

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, n_actions))


# Define the function to get a random starting position for the agent
def get_start_position():
    return (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))


# Define the function to take an action
def take_action(state, action):
<<<<<<<<<<<<<<  ✨ Codeium Command ⭐ >>>>>>>>>>>>>>>>
def take_action(state, action):
    """
    Takes an action in the grid environment.

    Args:
        state (tuple): The current state of the agent (x, y).
        action (str): The action to take (up, down, left, right).

    Returns:
        tuple: The new state of the agent (x, y).

    The agent's new state is determined by the action and the current state.
    If the action is 'up', the agent moves up one cell. If the action is 'down',
    the agent moves down one cell. If the action is 'left', the agent moves left
    one cell. If the action is 'right', the agent moves right one cell.
    The agent's position is bounded by the grid size.
    """
    if action == 'up':
        return max(state[0] - 1, 0), state[1]
    elif action == 'down':
        return min(state[0] + 1, grid_size - 1), state[1]
    elif action == 'left':
        return state[0], max(state[1] - 1, 0)
    elif action == 'right':
        return state[0], min(state[1] + 1, grid_size - 1)
<<<<<<<  e173f71d-b6ba-44d4-9c28-47e1b63301b3  >>>>>>>
    if action == 'up':
        return max(state[0] - 1, 0), state[1]
    elif action == 'down':
        return min(state[0] + 1, grid_size - 1), state[1]
    elif action == 'left':
        return state[0], max(state[1] - 1, 0)
    elif action == 'right':
        return state[0], min(state[1] + 1, grid_size - 1)


# Define the reward function
def get_reward(state):
    if state == goal:
        return 1  # Positive reward for reaching the goal
    elif state in obstacles:
        return -1  # Negative reward for hitting an obstacle
    else:
        return -0.1  # Small negative reward for each step


# Q-Learning algorithm
def train():
    for episode in range(n_episodes):
        state = get_start_position()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)  # Explore
            else:
                action = actions[np.argmax(Q[state[0], state[1]])]  # Exploit

            new_state = take_action(state, action)
            reward = get_reward(new_state)
            Q[state[0], state[1], actions.index(action)] = (1 - alpha) * Q[
                state[0], state[1], actions.index(action)] + alpha * (reward + gamma * np.max(
                Q[new_state[0], new_state[1]]))

            state = new_state

            if state == goal:
                done = True

            # Visualization during training process (optional, can be slow)
            if episode % 100 == 0 and episode != 0:
                simulate_agent_path()


# Visualizing the agent's path after training
def simulate_agent_path():
    state = get_start_position()
    path = [state]
    while state != goal:
        action = actions[np.argmax(Q[state[0], state[1]])]
        state = take_action(state, action)
        path.append(state)

    print("Path to goal:", path)

    # Plot the environment and the path
    grid = np.zeros((grid_size, grid_size))
    for obs in obstacles:
        grid[obs] = -1  # Obstacles

    grid[goal] = 2  # Goal
    for i, pos in enumerate(path):
        grid[pos] = i + 1  # Agent's path

    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.title("Learning Path of Agent")
    plt.colorbar()
    plt.show()


# Visualizing the agent's path after full training
def visualize_final_path():
    state = get_start_position()
    path = [state]
    while state != goal:
        action = actions[np.argmax(Q[state[0], state[1]])]
        state = take_action(state, action)
        path.append(state)

    print("Final Path to Goal:", path)

    # Plot the environment and the path
    grid = np.zeros((grid_size, grid_size))
    for obs in obstacles:
        grid[obs] = -1  # Obstacles

    grid[goal] = 2  # Goal
    for i, pos in enumerate(path):
        grid[pos] = i + 1  # Agent's path

    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.title("Final Path after Training")
    plt.colorbar()
    plt.show()


# Train the model and visualize the path after every 100 episodes
start_time = time.time()
train()
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

# After training, visualize the agent's final path to the goal
visualize_final_path()
