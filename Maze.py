import numpy as np
import random
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, width=21, height=21):
        self.width = width
        self.height = height
        self.maze = np.ones((height, width), dtype=int)
        self._generate_maze()

    def _generate_maze(self):
        self.maze[1][1] = 0
        self._carve_passages_from(1, 1)

    def _carve_passages_from(self, cx, cy):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (1 <= nx < self.width-1 and 1 <= ny < self.height-1):
                if self.maze[ny][nx] == 1:
                    self.maze[cy + dy//2][cx + dx//2] = 0
                    self.maze[ny][nx] = 0
                    self._carve_passages_from(nx, ny)

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self):
        self.player_pos = [1, 1]
        return self.player_pos

    def step(self, action):
        x, y = self.player_pos
        moves = {'up': (x, y - 1), 'down': (x, y + 1),
                 'left': (x - 1, y), 'right': (x + 1, y)}
        x_new, y_new = moves[action]
        if self.maze[y_new][x_new] == 0:
            self.player_pos = [x_new, y_new]
        reward = -1
        done = False
        if self.player_pos == [self.maze.shape[1]-2, self.maze.shape[0]-2]:
            reward = 100
            done = True
        return self.player_pos, reward, done

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            q_values = [self.get_q(state, a) for a in self.env.actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(self.env.actions, q_values)
                           if q == max_q]
            return random.choice(max_actions)

    def learn(self, state, action, reward, next_state):
        q_predict = self.get_q(state, action)
        q_target = reward + self.gamma * max(
            [self.get_q(next_state, a) for a in self.env.actions])
        self.q_table[(tuple(state), action)] = q_predict + \
            self.alpha * (q_target - q_predict)

def train_agent(env, agent, episodes=500):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step_count += 1
            if done:
                break
        if (episode+1) % 100 == 0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward}, "
                  f"Steps: {step_count}")

def run_agent(env, agent):
    state = env.reset()
    path = [state.copy()]
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        path.append(next_state.copy())
        state = next_state
        if done:
            break
    return path

def display_maze_with_path(maze, path, start_pos, goal_pos):
    maze_display = np.copy(maze)
    # values for visualization
    # 0: path (white)
    # 1: wall (black)
    # 2: agent's path (blue)
    # 3: start position (green)
    # 4: goal position (red)
    for pos in path:
        maze_display[pos[1]][pos[0]] = 2
    maze_display[start_pos[1]][start_pos[0]] = 3
    maze_display[goal_pos[1]][goal_pos[0]] = 4
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'black', 'blue', 'green', 'red'])
    plt.figure(figsize=(10, 10))
    plt.imshow(maze_display, cmap=cmap)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    maze_obj = Maze()
    # Define start and goal positions
    start_pos = [1, 1]
    goal_pos = [maze_obj.width-2, maze_obj.height-2]
    # Display the maze with start and goal positions
    display_maze_with_path(maze_obj.maze, [], start_pos, goal_pos)
    env = MazeEnv(maze_obj.maze)
    agent = QLearningAgent(env)
    train_agent(env, agent)
    path = run_agent(env, agent)
    # Visualize the maze with the path, start, and goal
    display_maze_with_path(maze_obj.maze, path, start_pos, goal_pos)
