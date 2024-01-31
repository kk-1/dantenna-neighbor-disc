import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation

#######################################################################################################
# Set the seeds
np.random.seed(2024)
random.seed(2024)


#######################################################################################################
# firmware instruction format:
# (type, duration)
# (1, t) - rotate same direction t times each time 1 degree
# (-1, t) - rotate opposite (initially anticlockwise?) dir t times each time 1 degree
# (0, t) - stop for t times
# (2, t) - rnd dir or stop t times
# (3, t) - rnd dir or stop pick time as rnd(MIN_RND_DURATION, upto t (t + MIN_RND_DURATION))
# Example firmware instruction: [(2, 90),  (3, 24), (2, 45), (-1, 45), (2, 45), (0, 45)]
#######################################################################################################
# Constants
# Weights for the random direction choice: Counterclockwise, Stop, Clockwise
DIRECTION_WEIGHTS = [1, 2, 1]  # Adjust weights as needed
MIN_RND_DURATION = 1  # Minimum random duration
theN = 3
theM = 3
GRID_SIZE = (theN, theM)  # n x m grid
TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID = 4 * theM * theN - 3 * theM - 3 * theN + 2
THETA = np.pi / 3  # 60 degrees sector angle
TIME_UNITS = 2000  # Total time units for simulation = frames
ROTATION_STEP = np.pi / 180  # 1 degree in radians
ANTENNA_RANGE = 1.5  # Range for each antenna
time_step = 0

all_pairs_so_far = set()
connected_reached = 0


#Adjacency matrix
adj_mtx = np.zeros([theN * theM, theN * theM], dtype = int)

#######################################################################################################



#######################################################################################################


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

#######################################################################################################

def is_spanning_tree(adj_mtx):
    """Check if the graph is a spanning tree."""
    n = len(adj_mtx)
    num_edges = sum(sum(row) for row in adj_mtx) // 2

    # Spanning tree must have n-1 edges
    if num_edges != n - 1:
        return False

    # Spanning tree must be connected
    return is_connected(adj_mtx, 0)




#######################################################################################################

def normalize_angle(angle):
        return angle % (2 * np.pi)


#######################################################################################################
class Antenna:
    def __init__(self, id, position, firmware):
        self.id = id
        self.position = position
        self.direction = random.uniform(0, 2 * np.pi)
        self.neighbors_so_far = set()
        self.num_of_pairings = 0
        self.firmware = firmware
        #print(self.id, self.firmware)
        self.program_counter = 0  # Counter for the duration of each instruction
        self.instruction_index = 0  # Index of the current instruction
        self.time_from_last_pairing = 0

        self.update_sector_angles()



    def rotate(self):
        global time_step

        if self.program_counter == 0:  # Time to switch to the next instruction
            instruction, max_duration = self.firmware[self.instruction_index]

            if instruction in [2, 3]:  # Random direction with weights
                directions = [-1, 0, 1]  # Counterclockwise, Stop, Clockwise
                self.rotation_direction = random.choices(directions, weights=DIRECTION_WEIGHTS, k=1)[0]

                if instruction == 3:  # Random duration for instruction 3
                    random_duration = random.randint(MIN_RND_DURATION, max_duration)
                    self.program_counter = random_duration
                else:
                    self.program_counter = max_duration
            else:
                self.rotation_direction = instruction
                self.program_counter = max_duration

        # Rotate based on the current instruction
        if self.rotation_direction in [1, -1]:
            self.direction += self.rotation_direction * ROTATION_STEP
            self.direction = self.normalize_angle(self.direction)
        # No action for self.rotation_direction == 0 (stop)

        self.program_counter -= 1
        if self.program_counter <= 0:
            self.instruction_index = (self.instruction_index + 1) % len(self.firmware)

        #print(time_step, "INS -> ",self.id, self.program_counter, self.firmware[self.instruction_index])
        self.update_sector_angles()



    def update_sector_angles(self):
        self.start_angle = self.normalize_angle(self.direction - THETA/2)
        self.end_angle = self.normalize_angle(self.direction + THETA/2)

    def draw_sector(self, ax):
        sector_vertices = self.calculate_sector_vertices()
        polygon = plt.Polygon(sector_vertices, color='skyblue', alpha=0.3)
        ax.add_patch(polygon)
        ax.plot(*self.position, 'bo')  # Antenna position
        ax.text(*self.position, f'{self.id}', color='black', ha='right', va='bottom')  # Antenna ID






    def calculate_sector_vertices(self, num_points=20):
        start_angle = self.normalize_angle(self.direction - THETA/2)
        end_angle = self.normalize_angle(self.direction + THETA/2)

        # Adjust for angle wraparound
        if end_angle < start_angle:
            end_angle += 2 * np.pi

        # Calculate the sector vertices in an anticlockwise direction
        angles = np.linspace(start_angle, end_angle, num_points)
        vertices = [self.position] + [self.position + ANTENNA_RANGE * np.array([np.cos(a), np.sin(a)]) for a in angles]

        return vertices


    @staticmethod
    def normalize_angle(angle):
        return angle % (2 * np.pi)

#######################################################################################################



def is_in_range_and_sector(point, center, direction, theta, rangex):
    # Calculate the relative position of the point from the center
    relative_position = point - center

    # Calculate the distance from the center to the point
    distance = np.linalg.norm(relative_position)

    # Check if the point is within the range of the antenna
    if distance > rangex:
        return False

    # Calculate the angle to the point from the center
    angle_to_point = np.arctan2(relative_position[1], relative_position[0])
    angle_to_point = normalize_angle(angle_to_point)

    # Normalize the direction of the antenna
    direction = normalize_angle(direction)

    # Calculate the start and end angles of the sector
    start_angle = normalize_angle(direction - theta/2)
    end_angle = normalize_angle(direction + theta/2)

    # Check if the angle to the point is within the sector
    if start_angle <= end_angle:
        return start_angle <= angle_to_point <= end_angle
    else:
        return angle_to_point >= start_angle or angle_to_point <= end_angle

#######################################################################################################

def antennas_see_each_other(antenna1, antenna2):
    return (is_in_range_and_sector(antenna1.position, antenna2.position, antenna2.direction, THETA, ANTENNA_RANGE) and
            is_in_range_and_sector(antenna2.position, antenna1.position, antenna1.direction, THETA, ANTENNA_RANGE))


#######################################################################################################

# Initialize antennas
firmware_program = [ (3, 50), (-1, 90),  (3, 50), (2, 45), (-1, 45), (2, 45), (0, 45)]
antennas = [Antenna(i, np.array([x, y]), firmware_program) for i, (x, y) in enumerate(np.ndindex(GRID_SIZE))]

#antennas = [Antenna(i, np.array([x, y])) for i, (x, y) in enumerate(np.ndindex(GRID_SIZE))]

def update(frame, antennas, ax):
    global time_step
    global all_pairs_so_far
    global adj_mtx
    global connected_reached

    time_step = frame
    nn = len(all_pairs_so_far)
    if (connected_reached == 0) and (nn != 0):
             if (is_connected(adj_mtx, list(all_pairs_so_far)[0][0])):
                print(f"Connected graph reached at Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.")
                print("All pairs of connected graph:",all_pairs_so_far)
                connected_reached = frame
                #return

    ax.clear()
    ax.set_xlim(-1, GRID_SIZE[0])
    ax.set_ylim(-1, GRID_SIZE[1])
    if (connected_reached == 0):
        ax.set_title(f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.\n Not Connected yet!")
    else:
         ax.set_title(f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.\n Connected graph reached at Time Unit: {connected_reached}!")

    #See stat while saving mp4

    print(f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.")

    for antenna in antennas:
        antenna.rotate()
        antenna.time_from_last_pairing = antenna.time_from_last_pairing + 1
        antenna.draw_sector(ax)



    if (nn == TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID):
        ax.set_title(f"Time Unit: {frame} - All neighbors discoverd!")
        return

    if (nn != 0):

        for pair in all_pairs_so_far:
            #print(pair)
            antenna1 = antennas[pair[0]]
            antenna2 = antennas[pair[1]]
            ax.plot([antenna1.position[0], antenna2.position[0]],
                        [antenna1.position[1], antenna2.position[1]],
                        color='green', linestyle='-', linewidth=1)


    discovered_pairs = set()
    for i, antenna1 in enumerate(antennas):
        for j, antenna2 in enumerate(antennas):
            if i < j and antennas_see_each_other(antenna1, antenna2):
                discovered_pairs.add((antenna1.id, antenna2.id))
                all_pairs_so_far.add((antenna1.id, antenna2.id))
                #print(antenna1.position, antenna2.position)
                adj_mtx[antenna1.id][antenna2.id] = 1
                adj_mtx[antenna2.id][antenna1.id] = 1


                #Add them to others n set
                antenna1.neighbors_so_far.add(antenna2.id)
                antenna2.neighbors_so_far.add(antenna1.id)

                #Update data
                antenna1.time_from_last_pairing = 0
                antenna2.time_from_last_pairing = 0

                antenna1.num_of_pairings = antenna1.num_of_pairings + 1
                antenna2.num_of_pairings = antenna2.num_of_pairings + 1


                #print(antenna1.id, antenna2.id)
                #print(antenna1.id,"n:", antenna1.neighbors_so_far)
                #print(antenna2.id,"n:", antenna2.neighbors_so_far)
                ax.plot(*antenna1.position, 'ro', ms=20)
                ax.plot(*antenna2.position, 'ro', ms=20)
                ax.plot([antenna1.position[0], antenna2.position[0]],
                        [antenna1.position[1], antenna2.position[1]],
                        color='red', linestyle='-', linewidth=2)
            else:
                ax.plot(*antenna1.position, 'bo')

        # Plot antenna IDs
        #ax.text(*antenna1.position, f'ID: {antenna1.id}', color='black', ha='center', va='center')

    for antenna in antennas:
        sector = antenna.calculate_sector_vertices()
        polygon = plt.Polygon(sector, color='skyblue', alpha=0.3)
        ax.add_patch(polygon)

    if discovered_pairs:
        # Count the number of discovered pairs
        number_of_pairs = len(discovered_pairs)
        # To see online stat:
        # print(f"Time Unit {frame}: Tot {nn} pairs. Current {number_of_pairs} pairs - {discovered_pairs}")

# https://stackoverflow.com/questions/22010586/how-to-adjust-animation-duration
# For the saved animation the duration is going to be frames * (1 / fps) (in seconds)
# For the display animation the duration is going to be frames * interval / 1000 (in seconds)

fig, ax = plt.subplots(figsize=(6, 6))
ani = animation.FuncAnimation(fig, update, frames=TIME_UNITS, fargs=(antennas, ax), interval=1)
# To save uncomment the following and comment  plt.show() - faster!!!
#ani.save("antenna" + str(theM) + "x" + str(theN) + ".mp4", fps=25.0, dpi=300)


# To show:
plt.show()

