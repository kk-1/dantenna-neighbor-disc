import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation


# QL parameters
ALPHA = 0.0
GAMMA = 0.0
EPSILON = 1.0

# Set the seeds
np.random.seed(2024)
random.seed(2024)

# Constants
N_STATE = 8  # Number of states and also the actions as the action means the next state number which can be the same state (action for STOP)

DIRECTION_WEIGHTS = [1, 2, 1]  # Adjust weights as needed
MIN_RND_DURATION = 1  # Minimum random duration


THETA = 2 * np.pi / N_STATE  # Arrange angle of antenna for covering the whole sector
TIME_UNITS = 2000  # Total time units for simulation = frames
MAX_ROTATION_STEP = np.pi / 18  # 10 degrees in radians
ANTENNA_RANGE = 1.5  # Range for each antenna
time_step = 0


all_pairs_so_far = set()
connected_reached = 0


def is_connected(adj_mtx, start_vertex):
    """Check if the graph is connected using DFS."""
    n = len(adj_mtx)
    visited = [False] * n
    stack = [start_vertex]

    while stack:
        vertex = stack.pop()
        if not visited[vertex]:
            visited[vertex] = True
            for i in range(n):
                if adj_mtx[vertex][i] == 1 and not visited[i]:
                    stack.append(i)

    return all(visited)


def normalize_angle(angle):
    return angle % (2 * np.pi)


def init():
    # (draw static background here if you need to)
    return []


class Antenna:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.direction = random.uniform(0, 2 * np.pi)
        self.target_direction = self.direction
        self.Q = {}  # Dictionary to store Q-values
        self.learning_rate = ALPHA
        self.discount_factor = GAMMA
        self.epsilon = EPSILON  # Exploration rate
        self.action_space = np.linspace(0, 2 * np.pi, N_STATE)  # Discrete action space
        self.neighbors_discovered = set()

        self.update_sector_angles()

    def choose_action(self, time_unit):
        # Get a RND action!!!!
        action = np.random.choice(self.action_space)
        # print(f"Time Unit {time_unit} - Antenna {self.id}: Random action chosen -> {np.degrees(action):.2f} degrees")
        return action

        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            # print(f"Time Unit {time_unit} - Antenna {self.id}: Random action chosen -> {np.degrees(action):.2f} degrees")
            return action
        else:
            action = self.get_best_action()
            # print(f"Time Unit {time_unit} - Antenna {self.id}: Best action chosen -> {np.degrees(action):.2f} degrees")
            return action

    def get_best_action(self):
        # Get the action with the highest Q-value for the current state
        current_state = self.discretize_direction(self.direction)
        q_values = [self.Q.get((current_state, a), 0) for a in self.action_space]
        max_q_value = max(q_values)
        best_actions = [
            a for a, q in zip(self.action_space, q_values) if q == max_q_value
        ]
        return random.choice(best_actions)

    def update_Q(self, old_direction, action, reward, new_direction, time_unit):
        old_state = self.discretize_direction(old_direction)
        new_state = self.discretize_direction(new_direction)
        future_rewards = [self.Q.get((new_state, a), 0) for a in self.action_space]
        best_future_reward = max(future_rewards)
        old_q_value = self.Q.get((old_state, action), 0)
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * best_future_reward - old_q_value
        )
        self.Q[(old_state, action)] = new_q_value
        # print(f"Time Unit {time_unit} - Antenna {self.id}: Q-table entries: {len(self.Q)} Updated Q-value for state {old_state}, {old_q_value:.2f} -> {new_state}, {new_q_value:.2f}")

    def discretize_direction(self, direction):
        # Discretize direction to simplify the state space
        n_bins = len(self.action_space)  # N_STATE
        bin_width = 2 * np.pi / n_bins
        return int(direction // bin_width)

    def discretize_action(self, action):
        # Discretize action to simplify the action space
        n_bins = N_STATE
        bin_width = 2 * np.pi / n_bins
        return int(action // bin_width)

    def rotate(self, time_unit):
        old_direction = self.direction
        self.target_direction = self.choose_action(time_unit)
        while self.direction != self.target_direction:
            self.rotate_towards_target()
            reward = self.check_reward()
            self.update_Q(
                old_direction, self.target_direction, reward, self.direction, time_unit
            )
            old_direction = self.direction

    # Teleport
    def rotate_towards_target(self):
        self.direction = self.target_direction
        self.update_sector_angles()

    def update_sector_angles(self):
        self.start_angle = self.normalize_angle(self.direction - THETA / 2)
        self.end_angle = self.normalize_angle(self.direction + THETA / 2)

    def check_reward(self):
        #
        # DEBUG for RND WLK no reward return 0
        # reward = 0
        return 0
        for antenna in antennas:
            if antenna.id != self.id and antennas_see_each_other(self, antenna):
                if antenna.id not in self.neighbors_discovered:
                    return 1  # Do not come here next time so
                    self.neighbors_discovered.add(antenna.id)
                else:
                    return -2  # Second time pairing is worse
        return 0

    def draw_sector(self, ax):
        sector_vertices = self.calculate_sector_vertices()
        polygon = plt.Polygon(sector_vertices, color="skyblue", alpha=0.3)
        ax.add_patch(polygon)
        ax.plot(*self.position, "bo")  # Antenna position
        ax.text(
            *self.position, f"{self.id}", color="black", ha="right", va="bottom"
        )  # Antenna ID

    def calculate_sector_vertices(self, num_points=20):
        start_angle = self.normalize_angle(self.direction - THETA / 2)
        end_angle = self.normalize_angle(self.direction + THETA / 2)

        # Adjust for angle wraparound
        if end_angle < start_angle:
            end_angle += 2 * np.pi

        # Calculate the sector vertices in an anticlockwise direction
        angles = np.linspace(start_angle, end_angle, num_points)
        vertices = [self.position] + [
            self.position + ANTENNA_RANGE * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ]

        return vertices

    @staticmethod
    def normalize_angle(angle):
        return angle % (2 * np.pi)


def is_in_range_and_sector(point, center, direction, theta, rangex):
    relative_position = point - center
    distance = np.linalg.norm(relative_position)
    if distance > rangex:
        return False

    angle_to_point = np.arctan2(relative_position[1], relative_position[0])
    angle_to_point = normalize_angle(angle_to_point)
    direction = normalize_angle(direction)

    start_angle = normalize_angle(direction - theta / 2)
    end_angle = normalize_angle(direction + theta / 2)

    if start_angle <= end_angle:
        return start_angle <= angle_to_point <= end_angle
    else:
        return angle_to_point >= start_angle or angle_to_point <= end_angle


def antennas_see_each_other(antenna1, antenna2):
    return is_in_range_and_sector(
        antenna1.position, antenna2.position, antenna2.direction, THETA, ANTENNA_RANGE
    ) and is_in_range_and_sector(
        antenna2.position, antenna1.position, antenna1.direction, THETA, ANTENNA_RANGE
    )


def update(frame, antennas, ax):
    global time_step
    global all_pairs_so_far
    global adj_mtx
    global connected_reached

    time_step = frame
    nn = len(all_pairs_so_far)
    if connected_reached == 0 and nn != 0:
        if is_connected(adj_mtx, list(all_pairs_so_far)[0][0]):
            # print(f"Connected graph reached at Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.")
            # print("All pairs of connected graph:", all_pairs_so_far)
            connected_reached = frame

    ax.clear()
    ax.set_xlim(-1, GRID_SIZE[0])
    ax.set_ylim(-1, GRID_SIZE[1])
    if connected_reached == 0:
        ax.set_title(
            f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.\n Not Connected yet!"
        )
    else:
        ax.set_title(
            f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.\n Connected graph reached at Time Unit: {connected_reached}!"
        )
        print(ALPHA, GAMMA, EPSILON, frame, nn, theN)
        # 4 by 4 grid corner antennas
        # print("Antenna 0 Q table:", antennas[0].Q)
        # print("Antenna 3 Q table:", antennas[3].Q)
        # print("Antenna 12 Q table:", antennas[12].Q)
        # print("Antenna 15 Q table:", antennas[15].Q)
        # print(np.linspace(0, 2 * np.pi, 8))
        # Uncomment the following for animation on screen
        # ani.event_source.stop()

        exit()

    # print(f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.")

    for antenna in antennas:
        antenna.rotate(time_unit=frame)
        antenna.draw_sector(ax)

    if nn == TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID:
        ax.set_title(f"Time Unit: {frame} - All neighbor pairs are discovered!")

        # Uncomment the following for animation on screen
        # ani.event_source.stop()
        exit()

    if nn != 0:
        for pair in all_pairs_so_far:
            antenna1 = antennas[pair[0]]
            antenna2 = antennas[pair[1]]
            ax.plot(
                [antenna1.position[0], antenna2.position[0]],
                [antenna1.position[1], antenna2.position[1]],
                color="green",
                linestyle="-",
                linewidth=1,
            )

    discovered_pairs = set()
    for i, antenna1 in enumerate(antennas):
        for j, antenna2 in enumerate(antennas):
            if i < j and antennas_see_each_other(antenna1, antenna2):
                discovered_pairs.add((antenna1.id, antenna2.id))
                all_pairs_so_far.add((antenna1.id, antenna2.id))
                adj_mtx[antenna1.id][antenna2.id] = 1
                adj_mtx[antenna2.id][antenna1.id] = 1

                antenna1.neighbors_discovered.add(antenna2.id)
                antenna2.neighbors_discovered.add(antenna1.id)

                ax.plot(*antenna1.position, "ro", ms=20)
                ax.plot(*antenna2.position, "ro", ms=20)
                ax.plot(
                    [antenna1.position[0], antenna2.position[0]],
                    [antenna1.position[1], antenna2.position[1]],
                    color="red",
                    linestyle="-",
                    linewidth=2,
                )
            else:
                ax.plot(*antenna1.position, "bo")

    for antenna in antennas:
        sector = antenna.calculate_sector_vertices()
        polygon = plt.Polygon(sector, color="skyblue", alpha=0.3)
        ax.add_patch(polygon)

    if discovered_pairs:
        number_of_pairs = len(discovered_pairs)
        # print(f"Time Unit {frame}: Tot {nn} pairs. Current {number_of_pairs} pairs - {discovered_pairs}")


try:
    theN = int(sys.argv[1])
    theM = int(sys.argv[1])

    # Set the seeds
    newseed = int(sys.argv[2])
    np.random.seed(newseed)
    random.seed(newseed)
except:
    e = sys.exc_info()[0]
    print(e)


GRID_SIZE = (theN, theM)  # n x m grid
TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID = 4 * theM * theN - 3 * theM - 3 * theN + 2
# Adjacency matrix
adj_mtx = np.zeros([theN * theM, theN * theM], dtype=int)

# Initialize antennas
antennas = [
    Antenna(i, np.array([x, y])) for i, (x, y) in enumerate(np.ndindex(GRID_SIZE))
]

fig, ax = plt.subplots(figsize=(6, 6))


###############################################
# The following is for non-plotting version
###############################################
for frame in range(TIME_UNITS):
    update(frame, antennas, ax)


###############################################
# The following is for plotting version
###############################################

# Uncomment the following for animation on screen
# ani = animation.FuncAnimation(fig, update, frames=TIME_UNITS, fargs=(antennas, ax), interval=1, repeat=False, init_func=init, blit=False)


# Uncomment the following for animation on screen
# plt.show()

# Uncomment the following for animation on screen
# del ani

# exit()

#
#
# ani = animation.FuncAnimation(fig, update, frames=TIME_UNITS, fargs=(antennas, ax), interval=1, repeat = False)
#
# # To show:
# plt.show()
# # plt.close(fig)
#
#
# # To save uncomment the following and comment  plt.show() - faster!!!
# # ani.save("antenna" + str(theM) + "x" + str(theN) + "-"+str(ALPHA) + "-" + str(GAMMA) + "-" + str(EPSILON) + ".mp4", fps=25.0, dpi=300)
# #ani.save('animation.mp4', writer='ffmpeg')
#
# del ani
#
# exit()
# # To save the animation, uncomment the following and comment plt.show()
# # ani.save("antenna_simulation.mp4", fps=25.0, dpi=300)
