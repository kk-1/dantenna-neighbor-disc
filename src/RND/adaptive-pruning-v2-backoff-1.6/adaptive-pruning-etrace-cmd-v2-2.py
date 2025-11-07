import argparse, sys, math, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Circle

# Sector pruning parameters
GVOID_THRESHOLD = 8  # tries with zero hits → declare void tune 6‑10 if desired
BASE_COOLDOWN = 60  # frames to stay disabled after first void 50 was original
BACKOFF = 1.6  # exponential factor for subsequent voids

REPEAT_PRUNE = 6  # after 6 repeated pairings → cut the sector


# QL parameters
ALPHA = 0.5
GAMMA = 0.7
EPSILON = 1.0  # Start with high exploration
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
DECAY_FACTOR = 0.5
IMPROVE_FACTOR = 2.0
EVALUATION_INTERVAL = 8  # Time steps to evaluate TD error
TD_ERROR_THRESHOLD = 0.0005  # Threshold for TD error to reduce epsilon

AVERAGE_EPSILON = EPSILON

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
MIN_SEP = 0.95 * ANTENNA_RANGE  # used only in random mode

time_step = 0
ts = 0


Sum_of_all_epsilon = 0

all_pairs_so_far = set()
connected_reached = 0

# ----- globals for the ε panel --------------------------------------------------

eps_txt = []  # text artists for ε
id_txt = []  # text artists for ID
cbar = None
# cmap = mpl.cm.get_cmap("YlOrRd")
cmap = mpl.colormaps.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbars = {}  # so we don’t create multiple colourbars
dots = []  # Circle artists (populated at first frame)


# ───────────────────── random‑topology helpers ──────────────────────────────


def grow_connected_layout(N, *, R, min_sep=0.0, seed=None):
    """
    Incremental generator of an R-connected random topology.

    Parameters
    ----------
    N        : number of antennas to place
    R        : antenna range
    min_sep  : minimum allowed distance between any two antennas (≤ R)
    seed     : int or numpy.random.Generator for reproducibility

    Returns
    -------
    xy : (N,2) array of antenna coordinates   – always connected
    """
    rng = np.random.default_rng(seed)
    min2 = min_sep**2

    # choose an arbitrary origin for the cluster
    xy = [rng.uniform(-R, R, size=2)]  # first node at random in disc
    while len(xy) < N:
        # pick a random “parent” already in the cluster
        parent = xy[rng.integers(len(xy))]

        # sample a child uniformly in the radius-R disc around the parent
        theta = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(min_sep, R)  # avoid being exactly on parent
        candidate = parent + r * np.array([np.cos(theta), np.sin(theta)])

        # local spacing test
        if min_sep and any(np.linalg.norm(candidate - p) ** 2 < min2 for p in xy):
            continue  # reject and resample child

        xy.append(candidate)

    return np.asarray(xy)


def _is_connected(adj):
    seen = [False] * len(adj)
    stack = [0]
    while stack:
        v = stack.pop()
        if seen[v]:
            continue
        seen[v] = True
        stack.extend(adj[v] - {v})
    return all(seen)


def _pairwise_dist(xy):
    d = xy[:, None, :] - xy[None, :, :]
    return np.einsum("ijk,ijk->ij", d, d, optimize=True)


def create_rnd_topology_MC(N, side, *, rng, R, min_sep, tries=10000):
    """Return N points in [0,side]² whose R‑disk graph is connected."""
    # MC algo
    rng = np.random.default_rng(rng)
    min2, R2 = min_sep**2, R**2
    for _ in range(tries):
        xy = rng.uniform(0, side, (N, 2))
        D2 = _pairwise_dist(xy)
        if min_sep > 0 and np.any((D2 < min2) & (D2 > 0)):
            continue
        adj = [set(np.flatnonzero(D2[i] <= R2)) for i in range(N)]
        if any(len(nbr) == 0 for nbr in adj):
            continue
        if _is_connected(adj):
            return xy
    raise RuntimeError(
        "Random layout: unable to satisfy connectivity with given N and side"
    )


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


def king_pairs(m, n):
    """
    Return an (E,2) array of zero-based vertex indices for the 8-neighbour
    (king-move) graph of an m×n lattice.  Each pair (u,v) appears exactly
    once with u < v.
    """
    idx = np.arange(m * n, dtype=int).reshape(m, n)

    # half-neighbourhood (to avoid duplicates): NW, N, NE, W
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    src, dst = [], []
    for dy, dx in offsets:
        a = idx[
            max(0, -dy) : m - max(0, dy),  # crop source window
            max(0, -dx) : n - max(0, dx),
        ]
        b = idx[
            max(0, dy) : m - max(0, -dy),  # crop dest window
            max(0, dx) : n - max(0, -dx),
        ]
        src.append(a.ravel())
        dst.append(b.ravel())

    return np.column_stack((np.concatenate(src), np.concatenate(dst)))


##################################################################################################


class Antenna:
    def __init__(self, id, position):
        self.id = id
        self.time_step = 0
        self.position = position
        self.action_space = np.linspace(
            0, 2 * np.pi, N_STATE, endpoint=False
        )  # Discrete action space
        # print("ID", self.id, "aspace:", self.action_space)

        # initialise direction to a VALID sector centre -----------------
        self.direction = float(random.choice(self.action_space))

        # self.direction = random.uniform(0, 2 * np.pi)
        self.target_direction = self.direction
        self.Q = {}  # Dictionary to store Q-values
        self.learning_rate = ALPHA
        self.discount_factor = GAMMA
        self.epsilon = EPSILON  # Exploration rate
        self.epsilon_min = EPSILON_MIN
        self.decay_factor = DECAY_FACTOR
        self.epsilon_max = EPSILON_MAX
        self.improve_factor = IMPROVE_FACTOR
        # self.action_space = np.linspace(0, 2 * np.pi, N_STATE)  # Discrete action space
        # self.action_space = np.linspace(
        #     0, 2 * np.pi, N_STATE, endpoint=False
        # )  # Discrete action space

        self.neighbors_discovered = set()
        self.previous_Q = {}  # Track previous Q-table
        self.rewards = []
        self.evaluation_interval = EVALUATION_INTERVAL
        self.td_error_threshold = TD_ERROR_THRESHOLD
        self.update_sector_angles()

        # --- local void‑sector pruning ---
        self.disabled_actions = set()  # sectors declared ‘void’ PERMANENT CUT
        self.visits = {a: 0 for a in self.action_space}
        self.hits = {a: 0 for a in self.action_space}
        self.VOID_THRESHOLD = GVOID_THRESHOLD

        #    for cool‑down pruning
        self.void_count = {}  # how many times each sector was declared void
        self.disabled_until = {}  # time‑step until which that sector is frozen

        self.repeats = {a: 0 for a in self.action_space}  # for sector cut

    ###############################################################
    # ε‑greedy action selection *with*   valid‑action masking
    ###############################################################
    def choose_action(self, time_unit):
        # 1) periodically adapt ε from TD‑error
        if time_unit % self.evaluation_interval == 0:
            self.update_epsilon(time_unit)

        # ONLY sectors whose cool‑down has expired
        action_pool = [
            a
            for a in self.action_space
            if a not in self.disabled_actions  # PERMANENT CUT
            and (time_unit >= self.disabled_until.get(a, -1))
        ]

        if not action_pool:  # safety: never empty
            action_pool = list(self.action_space)

        current_state = self.discretize_direction(self.direction)  # ← NEW

        # 2) exploration vs. exploitation
        if random.random() < self.epsilon:
            # purely exploratory
            return np.random.choice(action_pool)
        else:
            # exploit the best Q so far – but **state** must exist first!
            q_vals = np.array(
                [
                    self.Q.get((current_state, a), 0.0)  # ← FIX
                    for a in action_pool
                ]
            )

            # pick randomly among all maximisers
            best_idxs = np.flatnonzero(q_vals == q_vals.max())
            return action_pool[np.random.choice(best_idxs)]

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

        # action_pool = [a for a in self.action_space if a not in self.disabled_actions]
        action_pool = [
            a
            for a in self.action_space
            if a not in self.disabled_actions  # PERMANENT CUT
            and (time_unit >= self.disabled_until.get(a, -1))
        ]

        if not action_pool:
            action_pool = list(self.action_space)
        future_rewards = [self.Q.get((new_state, a), 0) for a in action_pool]

        best_future_reward = max(future_rewards)
        old_q_value = self.Q.get((old_state, action), 0)
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * best_future_reward - old_q_value
        )
        self.Q[(old_state, action)] = new_q_value

        self.visits[action] += 1
        if reward > 0:
            self.hits[action] += 1
            self.repeats[action] = 0  # reset repeat counter
        elif reward == -2:  # repairing the same edge
            self.repeats[action] += 1

        if (self.visits[action] >= GVOID_THRESHOLD and self.hits[action] == 0) or (
            self.repeats[action] >= REPEAT_PRUNE
        ):
            #  book‑keeping
            count = self.void_count.get(action, 0) + 1
            self.void_count[action] = count

            #  exponential back‑off
            cooldown = BASE_COOLDOWN * (BACKOFF ** (count - 1))
            self.disabled_until[action] = time_unit + cooldown  # ⟵ needs time!

            #  (re)‑initialise counters for the next evaluation window
            self.visits[action] = 0
            self.hits[action] = 0
            self.repeats[action] = 0

        # Track reward
        self.rewards.append(reward)
        if len(self.rewards) > self.evaluation_interval:
            self.rewards.pop(0)

        # Store current Q-table for future evaluation
        self.previous_Q[(old_state, action)] = old_q_value

    def update_epsilon(self, time_unit):
        # Calculate the average TD error
        td_errors = []
        for (state, action), q_value in self.Q.items():
            old_value = self.previous_Q.get((state, action), 0)
            td_errors.append(abs(q_value - old_value))

        avg_td_error = sum(td_errors) / len(td_errors) if td_errors else 0

        # Reduce epsilon if TD error is below threshold
        if avg_td_error < self.td_error_threshold:
            old = self.epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)
            # DEBUG
            # print(
            #     "Time:",
            #     time_unit,
            #     "Antenna",
            #     self.id,
            #     "epsilon decay:",
            #     old,
            #     "to",
            #     self.epsilon,
            # )
        else:
            old = self.epsilon
            self.epsilon = min(self.epsilon_max, self.epsilon * self.improve_factor)
            # DEBUG
            # print(
            #     "Time:",
            #     time_unit,
            #     "Antenna",
            #     self.id,
            #     "epsilon improve:",
            #     old,
            #     "to",
            #     self.epsilon,
            # )

        # if avg_td_error < self.td_error_threshold:
        #     old = self.epsilon
        #     self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)
        # DEBUG
        # print("Time:",time_unit,"Antenna",self.id,"epsilon update:",old, "to",self.epsilon)

    def discretize_direction(self, direction):
        # Discretize direction to simplify the state space
        # n_bins = len(self.action_space)  # N_STATE
        # bin_width = 2 * np.pi / n_bins
        direction = normalize_angle(direction)
        return int(direction // (2 * np.pi / len(self.action_space)))
        # return int(direction // bin_width)

    def rotate(self, time_unit):
        old_dir = self.direction
        action = self.choose_action(time_unit)  # decide where to look
        self.direction = action  # instant “teleport” rotation
        reward = self.check_reward()  # see if we paired
        self.update_Q(old_dir, action, reward, self.direction, time_unit)

    def update_sector_angles(self):
        self.start_angle = normalize_angle(self.direction - THETA / 2)
        self.end_angle = normalize_angle(self.direction + THETA / 2)

    def check_reward(self):
        global all_pairs_so_far, adj_mtx
        global time_step
        for antenna in antennas:
            if antenna.id != self.id and antennas_see_each_other(self, antenna):
                if antenna.id not in self.neighbors_discovered:
                    self.neighbors_discovered.add(antenna.id)
                    all_pairs_so_far.add(
                        (min(self.id, antenna.id), max(self.id, antenna.id))
                    )
                    adj_mtx[self.id, antenna.id] = 1
                    adj_mtx[antenna.id, self.id] = 1

                    # Plot red line for new pair
                    ax.plot(*self.position, "ro", ms=20)
                    ax.plot(*antenna.position, "ro", ms=20)
                    ax.plot(
                        [self.position[0], antenna.position[0]],
                        [self.position[1], antenna.position[1]],
                        color="red",
                        linestyle="-",
                        linewidth=2,
                    )

                    # Cut the sector

                    self.disabled_actions.add(self.direction)  # PERMANENT CUT
                    antenna.disabled_actions.add(antenna.direction)  # PERMANENT CUT
                    # DEBUG
                    # print(time_step,"Pairing:", self.id,"-", antenna.id,"Antenna",self.id,  "cuts",   self.direction, "--- Antenna",  antenna.id, "cuts", antenna.direction)

                    return 1  # New neighbor discovered
                else:
                    return -2  # Repeated pairing
        return 0

    def draw_sector_plain(self, ax):
        sector_vertices = self.calculate_sector_vertices()
        polygon = plt.Polygon(sector_vertices, color="skyblue", alpha=0.3)
        ax.add_patch(polygon)
        ax.plot(*self.position, "bo")  # Antenna position
        ax.text(
            *self.position, f"{self.id}", color="black", ha="right", va="bottom"
        )  # Antenna ID

    def draw_sector(self, ax, time_unit):
        """Draw all 8 sectors:
        • current beam  → green (opaque)
        • void cool-down → gray (transparent)
        • permanent cut  → red  (transparent)
        • untouched      → sky-blue (transparent)"""

        for a in self.action_space:
            verts = self.calculate_sector_vertices_for(a)
            if a == self.direction:  # current pointing
                colour, alpha = "limegreen", 0.6
            elif a in self.disabled_actions:  # permanently cut
                colour, alpha = "red", 0.3
            elif time_unit < self.disabled_until.get(a, -1):  # temp void
                colour, alpha = "blue", 0.2
            else:  # normal sector
                colour, alpha = "white", 0.2

            ax.add_patch(plt.Polygon(verts, color=colour, alpha=alpha, ec=None))

        # antenna body & id
        ax.plot(*self.position, "ko", ms=4)
        ax.text(*self.position, f"{self.id}", ha="right", va="bottom")

    def calculate_sector_vertices_for(self, sector_angle, n=12):
        start = normalize_angle(sector_angle - THETA / 2)
        end = normalize_angle(sector_angle + THETA / 2)
        if end < start:
            end += 2 * np.pi
        angs = np.linspace(start, end, n)
        return [self.position] + [
            self.position + ANTENNA_RANGE * np.array([np.cos(a), np.sin(a)])
            for a in angs
        ]

    def calculate_sector_vertices(self, num_points=20):
        start_angle = normalize_angle(self.direction - THETA / 2)
        end_angle = normalize_angle(self.direction + THETA / 2)

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


def antennas_see_each_other(antenna1, antenna2):
    return is_in_range_and_sector(
        antenna1.position, antenna2.position, antenna2.direction, THETA, ANTENNA_RANGE
    ) and is_in_range_and_sector(
        antenna2.position, antenna1.position, antenna1.direction, THETA, ANTENNA_RANGE
    )


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


def update(frame, antennas, ax):
    global time_step
    global all_pairs_so_far
    global adj_mtx
    global connected_reached
    global ts
    global Sum_of_all_epsilon
    global AVERAGE_EPSILON
    global etracef

    time_step = frame
    for antenna in antennas:
        Sum_of_all_epsilon = Sum_of_all_epsilon + antenna.epsilon
    ts = ts + 1
    # print(frame, end=", ")

    joined_eps = ", ".join(str(antenna.epsilon) for antenna in antennas)
    # print(joined_eps)
    etracef.write(str(frame) + ", " + joined_eps + "\n")
    #
    # for antenna in antennas:
    #     print(antenna.epsilon, end=", ")
    # print("\n")

    for antenna in antennas:
        antenna.time_step = frame  # Sync the time step for each antenna

    ax.clear()
    ax.set_xlim(XMIN, XMAX)  # keep the window frozen
    ax.set_ylim(YMIN, YMAX)

    # draw_eps_panel(frame, antennas)

    # pale-blue backdrop: every potential link in the R-disk graph
    for i, j in reachable_pairs:
        ax.plot(
            [antennas[i].position[0], antennas[j].position[0]],
            [antennas[i].position[1], antennas[j].position[1]],
            color="gray",
            linewidth=1,
            alpha=0.4,
        )

    # Draw sectors and pair lines
    for antenna in antennas:
        antenna.rotate(time_step)
        # DEBUG uncomment for colorful (color cut type)
        antenna.draw_sector(ax, time_step)
        # DEBUG uncomment for plain sectors
        # antenna.draw_sector_plain(ax)

    for a, b in all_pairs_so_far:
        ax.plot(
            [antennas[a].position[0], antennas[b].position[0]],
            [antennas[a].position[1], antennas[b].position[1]],
            "r-",
        )

    nn = len(all_pairs_so_far)
    if connected_reached == 0 and nn != 0:
        if is_connected(adj_mtx, list(all_pairs_so_far)[0][0]):
            connected_reached = frame
            # print("Connected! at frame:",frame, time_step)

    if connected_reached == 0:
        ax.set_title(
            f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.\n Not Connected yet!"
        )

    else:
        ax.set_title(
            f"Time Unit: {frame} -  {nn} edges of {TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID}: {round(100 * nn / TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID, 2)} % completed.\n Connected graph reached at Time Unit: {connected_reached}!"
        )
        # sumE = 0
        # for antenna in antennas:
        #     sumE = sumE + antenna.epsilon
        # print(ALPHA, GAMMA, round(sumE / (theN * theN), 2), frame, nn, theN)
        AVERAGE_EPSILON = round(Sum_of_all_epsilon / (ts * theN * theN), 2)
        print(
            ALPHA,
            GAMMA,
            EPSILON,
            round(Sum_of_all_epsilon / (ts * theN * theN), 2),
            frame,
            nn,
            theN,
            TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID,
        )

        # # print("***** Result Connected! at frame:",frame, time_step, ts)
        # # Uncomment the following for animation on screen
        # fig.canvas.draw()
        # # DEBUG
        # save_eps_snapshot(frame, "eps_heat_final")
        #
        # ani.event_source.stop()
        #
        # # DEBUG
        # print("Time, ts:", time_step, ts)
        # print("AVG E:", round(Sum_of_all_epsilon / (ts * theN * theN), 2))
        exit()


##########################################################################
# To animate epsilon values
##########################################################################


def draw_eps_panel(frame_idx: int, swarm):
    """Update (or create) the right-hand ε panel."""

    global dots, eps_txt, id_txt, cbar
    global AVERAGE_EPSILON
    global ts
    global Sum_of_all_epsilon

    AVERAGE_EPSILON = round(Sum_of_all_epsilon / (ts * theN * theN), 2)

    ax_eps.set_title(
        f"$\epsilon$ snapshot. AVG $\epsilon$= {AVERAGE_EPSILON}", fontsize=10
    )

    # create on first call ------------------------------------------------
    if frame_idx == 0:
        dots.clear()
        ax_eps.clear()
        ax_eps.set_aspect("equal")
        ax_eps.set_title(r"$\epsilon$ snapshot", fontsize=10)
        ax_eps.set_xticks([])
        ax_eps.set_yticks([])

        id_colour = "black"  # change if labels hard to read

        # -------------------------------- lattice and random coords ------
        coords = [ant.position for ant in swarm]

        # create circle + label for every antenna ------------------------
        for pos, ant in zip(coords, swarm):
            circ = Circle(pos, 0.5, ec="black", lw=0.25, zorder=2)
            ax_eps.add_patch(circ)
            dots.append(circ)

            t_id = ax_eps.text(
                *pos,
                str(ant.id),
                ha="center",
                va="center",
                fontsize=7,
                color=id_colour,
                zorder=3,
            )

            id_txt.append(t_id)

            t_ep = ax_eps.text(
                pos[0],
                pos[1] - 0.25,  # small downward shift
                f"{ant.epsilon:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                zorder=3,
            )
            eps_txt.append(t_ep)

        # colour-bar (once)
        if "eps" not in cbars:
            cbars["eps"] = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax_eps,
                fraction=0.06,
                pad=0.05,  # <- wider pad
                label=r"$\epsilon$",
            )

        # tight limits
        xs, ys = np.vstack(coords).T
        pad = 0.6
        ax_eps.set_xlim(xs.min() - pad, xs.max() + pad)
        ax_eps.set_ylim(ys.min() - pad, ys.max() + pad)

    # ── every frame → update colours *and* ε-texts ───────────────────
    for circ, txt, ant in zip(dots, eps_txt, swarm):
        circ.set_facecolor(cmap(norm(ant.epsilon)))
        txt.set_text(f"{ant.epsilon:.2f}")

    # # update colours every frame -----------------------------------------
    # for circ, ant in zip(dots, swarm):
    #     circ.set_facecolor(cmap(norm(ant.epsilon)))


def save_eps_snapshot(idx, prefix="eps_heat"):
    draw_eps_panel(idx, antennas)
    fig.savefig(f"{prefix}-{idx}.png", dpi=300, bbox_inches="tight")
    print("saved", f"{prefix}-{idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# CLI handling — keep legacy positional args, add optional random section
# ─────────────────────────────────────────────────────────────────────────────

try:
    # legacy 5 args
    ALPHA = float(sys.argv[1])
    GAMMA = float(sys.argv[2])
    EPSILON = float(sys.argv[3])
    theN = theM = int(sys.argv[4])  # grid size N×N OR area width
    seed = int(sys.argv[5])
    etrace_fname = sys.argv[6]
    # DEBUG
    # print("Fname is", etrace_fname)
    etracef = open(etrace_fname, "a")
    header = ["time"] + [f"a{i}" for i in range(1, theN * theN + 1)]
    header_txt = ", ".join(header)
    # DEBUG
    # print(header_txt)
    etracef.write(header_txt + "\n")

    random_flag = len(sys.argv) > 7 and sys.argv[7].lower() == "random"
except (IndexError, ValueError):
    print("Usage: α γ ε N seed  [random]")
    sys.exit(1)


np.random.seed(seed)
random.seed(seed)
NUM_NODES = theN * theN
GRID_SIZE = (theN, theM)

# ─────────────────────────────────────────────────────────────────────────────
# Topology creation
# ─────────────────────────────────────────────────────────────────────────────

if random_flag:
    SIDE = 2.5 * theN * ANTENNA_RANGE
    NUM_NODES = theN * theN

    xy = grow_connected_layout(NUM_NODES, R=ANTENNA_RANGE, min_sep=MIN_SEP, seed=seed)
    SIDE = 1.1 * np.ptp(xy, axis=0).max()  # bounding box for plotting
    # xy = create_rnd_topology_MC(
    #      NUM_NODES, SIDE, rng=seed, R=ANTENNA_RANGE, min_sep=MIN_SEP
    #  )
    NODE_COUNT = NUM_NODES

else:
    SIDE = theN
    xy = np.array([[x, y] for y in range(theN) for x in range(theN)], float)
    reachable_pairs = king_pairs(theN, theN)
    NODE_COUNT = theN * theN


# bounding box for plotting
XMIN, XMAX = xy[:, 0].min() - 1, xy[:, 0].max() + 1
YMIN, YMAX = xy[:, 1].min() - 1, xy[:, 1].max() + 1

# ------- adjacency and agent objects ---------------------------------------
adj_mtx = np.zeros((NODE_COUNT, NODE_COUNT), int)
# Antenna class must already be defined
antennas = [Antenna(i, xy[i]) for i in range(NODE_COUNT)]


if random_flag:
    # ----- compute all potentially reachable pairs once -----------------
    D2 = _pairwise_dist(xy)
    reach_mask = np.triu(D2 <= ANTENNA_RANGE**2, k=1)  # upper-triangle
    reachable_pairs = np.column_stack(np.where(reach_mask))  # list of (i,j)
    TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID = int(reach_mask.sum())
else:
    # original grid formula (queen’s graph)
    TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID = 4 * theN * theN - 3 * theN - 3 * theN + 2


# print("Total possible edges:", TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID)

##################################################################################################


# fig, ax = plt.subplots(figsize=(10, 10))

fig, (ax, ax_eps) = plt.subplots(
    1, 2, figsize=(16, 12), gridspec_kw={"width_ratios": [3, 2]}
)
fig.tight_layout()


ax.set_aspect("equal", adjustable="box")
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)


###############################################
# The following is for non-plotting version
###############################################
for frame in range(TIME_UNITS):
    update(frame, antennas, ax)


# print("Time limit reached! Not connected graph!",ALPHA, GAMMA, EPSILON, frame, theN)

###############################################
# The following is for plotting version
###############################################

# Uncomment the following for animation on screen

# print("Total possible edges:", TOTAL_QUEEN_EDGES_FOR_CONNECTED_GRID)
#
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=TIME_UNITS,
#     fargs=(antennas, ax),
#     interval=0,
#     repeat=False,
#     init_func=init,
#     blit=False,
# )


# Uncomment the following for animation on screen
# plt.show()

# Uncomment the following for animation on screen
# del ani


exit()
