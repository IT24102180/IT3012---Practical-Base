import random
import tkinter as tk
from collections import deque
import heapq


# ============================================================
# ENVIRONMENT
# ============================================================

class VisualGridHuntGame:
    """Partially observable Pacman-style grid environment."""

    def __init__(
        self,
        width=10,
        height=10,
        num_food=10,
        num_opponents=2,
        custom_walls=None
    ):

        self.width = width
        self.height = height

        # Agent starting position
        self.agent_pos = [0, 0]

        # Walls
        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            self.walls = {
                (2, 2),
                (2, 3),
                (5, 5),
                (6, 5),
                (3, 7)
            }

        # ----------------------------------------------------
        # Generate food
        # ----------------------------------------------------

        self.food_positions = set()

        while len(self.food_positions) < num_food:

            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)

            if (
                (fx, fy) != (0, 0)
                and (fx, fy) not in self.walls
            ):
                self.food_positions.add((fx, fy))

        # ----------------------------------------------------
        # Generate opponents
        # ----------------------------------------------------

        self.opponents = []

        while len(self.opponents) < num_opponents:

            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)

            if (
                (ox, oy) != (0, 0)
                and (ox, oy) not in self.walls
                and (ox, oy) not in self.food_positions
            ):
                self.opponents.append([ox, oy])

        # Game information
        self.score = 0
        self.steps = 0
        self.collision = False

    # ========================================================
    # PERCEPT
    # ========================================================

    def get_percept(self):

        x, y = self.agent_pos

        return {
            "wall_ahead": self.check_wall_ahead(),
            "food_here": (x, y) in self.food_positions,
            "opponent_nearby": self.check_opponent_nearby(),
            "score": self.score,

            # ------------------------------------------------
            # GLOBAL STATE FOR SEARCH AGENT
            # ------------------------------------------------

            "grid_size": (self.width, self.height),
            "walls": list(self.walls),
            "all_food": list(self.food_positions)
        }

    # ========================================================
    # CHECK WALL
    # ========================================================

    def check_wall_ahead(self):

        possible_moves = [
            (self.agent_pos[0], self.agent_pos[1] + 1),
            (self.agent_pos[0], self.agent_pos[1] - 1),
            (self.agent_pos[0] - 1, self.agent_pos[1]),
            (self.agent_pos[0] + 1, self.agent_pos[1])
        ]

        return any(
            pos in self.walls
            or pos[0] < 0
            or pos[0] >= self.width
            or pos[1] < 0
            or pos[1] >= self.height
            for pos in possible_moves
        )

    # ========================================================
    # CHECK OPPONENT
    # ========================================================

    def check_opponent_nearby(self):

        for op in self.opponents:

            distance = (
                abs(op[0] - self.agent_pos[0])
                + abs(op[1] - self.agent_pos[1])
            )

            if distance <= 1:
                return True

        return False

    # ========================================================
    # EXECUTE ACTION
    # ========================================================

    def execute_action(self, action):

        self.steps += 1

        new_pos = list(self.agent_pos)

        # ----------------------------------------------------
        # Move agent
        # ----------------------------------------------------

        if action == "Up":
            new_pos[1] += 1

        elif action == "Down":
            new_pos[1] -= 1

        elif action == "Left":
            new_pos[0] -= 1

        elif action == "Right":
            new_pos[0] += 1

        # ----------------------------------------------------
        # Check wall / boundary
        # ----------------------------------------------------

        if (
            tuple(new_pos) in self.walls
            or new_pos[0] < 0
            or new_pos[0] >= self.width
            or new_pos[1] < 0
            or new_pos[1] >= self.height
        ):

            self.score -= 5

        else:
            self.agent_pos = new_pos

        # ----------------------------------------------------
        # Collect food
        # ----------------------------------------------------

        position = tuple(self.agent_pos)

        if position in self.food_positions:

            self.food_positions.remove(position)

            self.score += 20

        # ----------------------------------------------------
        # Move opponents
        # ----------------------------------------------------

        for op in self.opponents:

            move = random.choice(
                ["Up", "Down", "Left", "Right", "Stay"]
            )

            if move == "Up" and op[1] < self.height - 1:
                op[1] += 1

            elif move == "Down" and op[1] > 0:
                op[1] -= 1

            elif move == "Left" and op[0] > 0:
                op[0] -= 1

            elif move == "Right" and op[0] < self.width - 1:
                op[0] += 1

            # ------------------------------------------------
            # Collision
            # ------------------------------------------------

            if op == self.agent_pos:

                self.score -= 50
                self.collision = True

    # ========================================================
    # GAME OVER
    # ========================================================

    def is_done(self):

        return (
            len(self.food_positions) == 0
            or self.steps >= 60
            or self.collision
        )


# ============================================================
# SEARCH AGENT
# ============================================================

class SearchAgent:
    """
    Goal-Based / Planning Agent.

    Supports:
        BFS - Breadth-First Search
        DFS - Depth-First Search
        UCS - Uniform-Cost Search
    """

    def __init__(self):

        # Complete offline plan
        self.plan = []

        # Default search algorithm
        self.active_algo = "BFS"

    # ========================================================
    # GET VALID NEIGHBOURS
    # ========================================================

    def get_neighbors(self, state, grid_size, walls):

        width, height = grid_size

        x, y = state

        possible_moves = [
            ("Up", (x, y + 1)),
            ("Down", (x, y - 1)),
            ("Left", (x - 1, y)),
            ("Right", (x + 1, y))
        ]

        neighbors = []

        for action, new_state in possible_moves:

            nx, ny = new_state

            # Check boundaries
            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Check wall
            if new_state in walls:
                continue

            neighbors.append(
                (action, new_state)
            )

        return neighbors

    # ========================================================
    # BFS
    # ========================================================

    def bfs_search(
        self,
        start,
        goal,
        grid_size,
        walls
    ):
        """
        Breadth-First Search.

        Uses FIFO queue.
        Explores shallowest nodes first.
        """

        frontier = deque()

        # Each item:
        # (current_state, path)
        frontier.append(
            (start, [])
        )

        # Reached set prevents repeated states
        reached = {start}

        while frontier:

            current, path = frontier.popleft()

            # Goal test
            if current == goal:
                return path

            # Expand current node
            for action, next_state in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                if next_state not in reached:

                    reached.add(next_state)

                    new_path = path + [action]

                    frontier.append(
                        (next_state, new_path)
                    )

        # No path found
        return []

    # ========================================================
    # DFS
    # ========================================================

    def dfs_search(
        self,
        start,
        goal,
        grid_size,
        walls
    ):
        """
        Depth-First Search.

        Uses LIFO stack.
        Explores deepest nodes first.
        """

        frontier = []

        frontier.append(
            (start, [])
        )

        # Reached set prevents loops
        reached = {start}

        while frontier:

            current, path = frontier.pop()

            # Goal test
            if current == goal:
                return path

            # Expand current node
            for action, next_state in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                if next_state not in reached:

                    reached.add(next_state)

                    new_path = path + [action]

                    frontier.append(
                        (next_state, new_path)
                    )

        # No path found
        return []

    # ========================================================
    # UCS
    # ========================================================

    def ucs_search(
        self,
        start,
        goal,
        grid_size,
        walls
    ):
        """
        Uniform-Cost Search.

        Uses a priority queue ordered by g(n),
        the total path cost.
        """

        frontier = []

        counter = 0

        # (cost, counter, state, path)
        heapq.heappush(
            frontier,
            (0, counter, start, [])
        )

        # Best known cost for each state
        reached = {
            start: 0
        }

        while frontier:

            cost, _, current, path = heapq.heappop(
                frontier
            )

            # Goal test
            if current == goal:
                return path

            # Expand current node
            for action, next_state in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                new_cost = cost + 1

                # Add state if this is the cheapest path
                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):

                    reached[next_state] = new_cost

                    counter += 1

                    new_path = path + [action]

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            next_state,
                            new_path
                        )
                    )

        # No path found
        return []

    # ========================================================
    # FIND CLOSEST FOOD
    # ========================================================

    def find_closest_food(
        self,
        start,
        food_positions,
        grid_size,
        walls
    ):
        """
        Find the closest reachable food.

        BFS is used to determine the shortest route to
        each available food pellet.
        """

        best_path = None
        best_food = None

        for food in food_positions:

            path = self.bfs_search(
                start,
                food,
                grid_size,
                walls
            )

            # Empty path can mean start == food.
            if start == food:
                return food, []

            # Ignore unreachable food
            if not path:
                continue

            # Keep shortest path
            if (
                best_path is None
                or len(path) < len(best_path)
            ):

                best_path = path
                best_food = food

        return best_food, best_path

    # ========================================================
    # SEARCH USING SELECTED ALGORITHM
    # ========================================================

    def search(
        self,
        start,
        goal,
        grid_size,
        walls
    ):

        if self.active_algo == "BFS":

            return self.bfs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == "DFS":

            return self.dfs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == "UCS":

            return self.ucs_search(
                start,
                goal,
                grid_size,
                walls
            )

        else:

            print(
                "Unknown algorithm:",
                self.active_algo
            )

            return []

    # ========================================================
    # SENSE AND ACT
    # ========================================================

    def sense_and_act(self, percept):

        # ----------------------------------------------------
        # If there is no existing plan,
        # create a new plan.
        # ----------------------------------------------------

        if not self.plan:

            # Current position
            current_position = tuple(
                percept.get(
                    "position",
                    (0, 0)
                )
            )

            # Get environment information
            grid_size = percept["grid_size"]

            walls = set(
                tuple(wall)
                for wall in percept["walls"]
            )

            all_food = [
                tuple(food)
                for food in percept["all_food"]
            ]

            # ------------------------------------------------
            # Find closest food
            # ------------------------------------------------

            closest_food, _ = self.find_closest_food(
                current_position,
                all_food,
                grid_size,
                walls
            )

            # ------------------------------------------------
            # Search for selected food
            # ------------------------------------------------

            if closest_food is not None:

                self.plan = self.search(
                    current_position,
                    closest_food,
                    grid_size,
                    walls
                )

        # ----------------------------------------------------
        # Execute first action in offline plan
        # ----------------------------------------------------

        if self.plan:

            return self.plan.pop(0)

        # ----------------------------------------------------
        # If no plan exists, choose a random action
        # ----------------------------------------------------

        return random.choice(
            ["Up", "Down", "Left", "Right"]
        )


# ============================================================
# GUI
# ============================================================

class GridGameGUI:

    def __init__(
        self,
        root,
        width=10,
        height=10,
        num_food=12,
        num_opponents=2,
        walls=None,
        algorithm="BFS"
    ):

        self.root = root

        self.root.title(
            "IT3012 - Practical 03 - Uninformed Search"
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        self.env = VisualGridHuntGame(
            width,
            height,
            num_food,
            num_opponents,
            walls
        )

        # ----------------------------------------------------
        # Search Agent
        # ----------------------------------------------------

        self.agent = SearchAgent()

        self.agent.active_algo = algorithm

        # ----------------------------------------------------
        # Canvas size
        # ----------------------------------------------------

        max_canvas_dim = 500

        self.cell_size = max(
            20,
            min(
                max_canvas_dim // width,
                max_canvas_dim // height
            )
        )

        self.canvas = tk.Canvas(
            root,
            width=width * self.cell_size,
            height=height * self.cell_size,
            bg="white"
        )

        self.canvas.pack()

        # ----------------------------------------------------
        # Status label
        # ----------------------------------------------------

        self.label = tk.Label(
            root,
            text=f"Algorithm: {algorithm} | Score: 0 | Steps: 0",
            font=("Arial", 14)
        )

        self.label.pack()

        # ----------------------------------------------------
        # Algorithm selection
        # ----------------------------------------------------

        self.algorithm_frame = tk.Frame(root)

        self.algorithm_frame.pack(
            pady=5
        )

        self.bfs_button = tk.Button(
            self.algorithm_frame,
            text="BFS",
            command=lambda: self.change_algorithm("BFS")
        )

        self.bfs_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.dfs_button = tk.Button(
            self.algorithm_frame,
            text="DFS",
            command=lambda: self.change_algorithm("DFS")
        )

        self.dfs_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.ucs_button = tk.Button(
            self.algorithm_frame,
            text="UCS",
            command=lambda: self.change_algorithm("UCS")
        )

        self.ucs_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------
        # Start button
        # ----------------------------------------------------

        self.btn = tk.Button(
            root,
            text="Start Simulation",
            command=self.run_loop
        )

        self.btn.pack(
            pady=5
        )

        # ----------------------------------------------------
        # Draw initial grid
        # ----------------------------------------------------

        self.draw_grid()

    # ========================================================
    # CHANGE ALGORITHM
    # ========================================================

    def change_algorithm(self, algorithm):

        self.agent.active_algo = algorithm

        # Clear existing plan so the new algorithm
        # is used for the next search.
        self.agent.plan = []

        self.label.config(
            text=(
                f"Algorithm: {algorithm} | "
                f"Score: {self.env.score} | "
                f"Steps: {self.env.steps}"
            )
        )

    # ========================================================
    # DRAW GRID
    # ========================================================

    def draw_grid(self):

        self.canvas.delete("all")

        # ----------------------------------------------------
        # Draw grid cells
        # ----------------------------------------------------

        for x in range(self.env.width):

            for y in range(self.env.height):

                x1 = x * self.cell_size

                y1 = (
                    self.env.height - 1 - y
                ) * self.cell_size

                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if (x, y) in self.env.walls:

                    color = "#64748b"

                else:

                    color = "#f1f5f9"

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=color,
                    outline="#cbd5e1"
                )

        # ----------------------------------------------------
        # Draw food
        # ----------------------------------------------------

        for fx, fy in self.env.food_positions:

            self.canvas.create_oval(
                fx * self.cell_size + 10,
                (
                    self.env.height - 1 - fy
                ) * self.cell_size + 10,

                fx * self.cell_size + 30,
                (
                    self.env.height - 1 - fy
                ) * self.cell_size + 30,

                fill="orange"
            )

        # ----------------------------------------------------
        # Draw opponents
        # ----------------------------------------------------

        for ox, oy in self.env.opponents:

            self.canvas.create_rectangle(
                ox * self.cell_size + 5,
                (
                    self.env.height - 1 - oy
                ) * self.cell_size + 5,

                ox * self.cell_size + 25,
                (
                    self.env.height - 1 - oy
                ) * self.cell_size + 25,

                fill="red"
            )

        # ----------------------------------------------------
        # Draw agent
        # ----------------------------------------------------

        ax, ay = self.env.agent_pos

        self.canvas.create_oval(
            ax * self.cell_size + 5,
            (
                self.env.height - 1 - ay
            ) * self.cell_size + 5,

            ax * self.cell_size + 30,
            (
                self.env.height - 1 - ay
            ) * self.cell_size + 30,

            fill="blue"
        )

    # ========================================================
    # RUN SIMULATION
    # ========================================================

    def run_loop(self):

        self.btn.config(
            state="disabled"
        )

        # Disable algorithm buttons
        self.bfs_button.config(
            state="disabled"
        )

        self.dfs_button.config(
            state="disabled"
        )

        self.ucs_button.config(
            state="disabled"
        )

        def step():

            if not self.env.is_done():

                # ------------------------------------------------
                # Get percept
                # ------------------------------------------------

                percept = self.env.get_percept()

                # ------------------------------------------------
                # Add current position to percept
                # ------------------------------------------------

                percept["position"] = tuple(
                    self.env.agent_pos
                )

                # ------------------------------------------------
                # Agent chooses action
                # ------------------------------------------------

                action = self.agent.sense_and_act(
                    percept
                )

                # ------------------------------------------------
                # Execute action
                # ------------------------------------------------

                self.env.execute_action(
                    action
                )

                # ------------------------------------------------
                # Redraw
                # ------------------------------------------------

                self.draw_grid()

                # ------------------------------------------------
                # Update status
                # ------------------------------------------------

                self.label.config(
                    text=(
                        f"Algorithm: "
                        f"{self.agent.active_algo} | "
                        f"Score: {self.env.score} | "
                        f"Steps: {self.env.steps} | "
                        f"Action: {action} | "
                        f"Plan remaining: "
                        f"{len(self.agent.plan)}"
                    )
                )

                # ------------------------------------------------
                # Next step
                # ------------------------------------------------

                self.root.after(
                    300,
                    step
                )

            else:

                self.label.config(
                    text=(
                        f"Game Over! | "
                        f"Algorithm: "
                        f"{self.agent.active_algo} | "
                        f"Score: {self.env.score} | "
                        f"Steps: {self.env.steps}"
                    )
                )

                self.btn.config(
                    state="normal"
                )

                self.bfs_button.config(
                    state="normal"
                )

                self.dfs_button.config(
                    state="normal"
                )

                self.ucs_button.config(
                    state="normal"
                )

        step()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = GridGameGUI(
        root,
        width=12,
        height=12,
        num_food=15,
        num_opponents=0,
        algorithm="BFS"
    )

    root.mainloop()