from collections import deque
import heapq
import math


class SearchAgent:
    """
    Search Agent for IT3012 Practicals 03 and 04.

    Supports:
        - BFS
        - DFS
        - UCS
        - A*
    """

    def __init__(self):
        self.plan = []
        self.active_algo = "BFS"

    # ==========================================================
    # GET NEIGHBORS
    # ==========================================================

    def get_neighbors(self, state, grid_size, walls):
        """
        Return valid neighboring states and the actions required
        to reach them.

        Movement:
            Up    -> y + 1
            Down  -> y - 1
            Left  -> x - 1
            Right -> x + 1
        """

        x, y = state
        width, height = grid_size

        possible_moves = [
            ((x, y + 1), "Up"),
            ((x, y - 1), "Down"),
            ((x - 1, y), "Left"),
            ((x + 1, y), "Right")
        ]

        neighbors = []

        for position, action in possible_moves:

            nx, ny = position

            # Check boundaries
            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Check walls
            if position in walls:
                continue

            neighbors.append((position, action))

        return neighbors

    # ==========================================================
    # BFS SEARCH
    # ==========================================================

    def bfs_search(self, start, goal, grid_size, walls):
        """
        Breadth-First Search.

        Returns:
            List of actions from start to goal.
        """

        queue = deque()

        # state, path
        queue.append((start, []))

        reached = {start}

        while queue:

            current, path = queue.popleft()

            # Goal test
            if current == goal:
                return path

            # Expand node
            for neighbor, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                if neighbor not in reached:

                    reached.add(neighbor)

                    new_path = path + [action]

                    queue.append(
                        (neighbor, new_path)
                    )

        return []

    # ==========================================================
    # DFS SEARCH
    # ==========================================================

    def dfs_search(self, start, goal, grid_size, walls):
        """
        Depth-First Search.

        Returns:
            List of actions from start to goal.
        """

        stack = []

        # state, path
        stack.append((start, []))

        reached = {start}

        while stack:

            current, path = stack.pop()

            # Goal test
            if current == goal:
                return path

            # Expand node
            for neighbor, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                if neighbor not in reached:

                    reached.add(neighbor)

                    new_path = path + [action]

                    stack.append(
                        (neighbor, new_path)
                    )

        return []

    # ==========================================================
    # UCS SEARCH
    # ==========================================================

    def ucs_search(self, start, goal, grid_size, walls):
        """
        Uniform Cost Search.

        Each movement has a cost of 1.

        Returns:
            List of actions from start to goal.
        """

        frontier = []

        counter = 0

        # cost, counter, state, path
        heapq.heappush(
            frontier,
            (0, counter, start, [])
        )

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

            # Expand node
            for neighbor, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                new_cost = cost + 1

                # If this is a cheaper path
                if (
                    neighbor not in reached
                    or new_cost < reached[neighbor]
                ):

                    reached[neighbor] = new_cost

                    counter += 1

                    new_path = path + [action]

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            neighbor,
                            new_path
                        )
                    )

        return []

    # ==========================================================
    # MANHATTAN DISTANCE
    # ==========================================================

    def manhattan_distance(self, pos, goal):
        """
        Calculate Manhattan distance between two positions.

        Formula:
            |x1 - x2| + |y1 - y2|
        """

        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    # ==========================================================
    # EUCLIDEAN DISTANCE
    # ==========================================================

    def euclidean_distance(self, pos, goal):
        """
        Calculate Euclidean distance between two positions.

        Formula:
            sqrt((x1 - x2)^2 + (y1 - y2)^2)
        """

        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )

    # ==========================================================
    # A* SEARCH
    # ==========================================================

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type="manhattan"
    ):
        """
        A* Search.

        f(n) = g(n) + h(n)

        g(n):
            Cost from start to current node.

        h(n):
            Estimated cost from current node to goal.

        heuristic_type:
            "manhattan" or "euclidean"

        Returns:
            List of actions from start to goal.
        """

        # Select heuristic
        if heuristic_type.lower() == "euclidean":
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        frontier = []

        # Starting cost
        g_cost = 0

        h_cost = heuristic(
            start_pos,
            goal_pos
        )

        f_cost = g_cost + h_cost

        counter = 0

        # ------------------------------------------------------
        # Priority queue item
        #
        # f_cost, g_cost, counter, current_pos, path_taken
        # ------------------------------------------------------

        heapq.heappush(
            frontier,
            (
                f_cost,
                g_cost,
                counter,
                start_pos,
                []
            )
        )

        # Best known g-cost for each state
        reached_states = {
            start_pos: 0
        }

        while frontier:

            (
                current_f,
                current_g,
                _,
                current_pos,
                path_taken
            ) = heapq.heappop(frontier)

            # Ignore outdated queue entries
            if (
                current_pos in reached_states
                and current_g > reached_states[current_pos]
            ):
                continue

            # --------------------------------------------------
            # Goal test
            # --------------------------------------------------

            if current_pos == goal_pos:
                return path_taken

            # --------------------------------------------------
            # Expand current node
            # --------------------------------------------------

            for neighbor, action in self.get_neighbors(
                current_pos,
                grid_size,
                walls
            ):

                # Every movement costs 1
                g_new = current_g + 1

                # Heuristic estimate
                h_new = heuristic(
                    neighbor,
                    goal_pos
                )

                # Total estimated cost
                f_new = g_new + h_new

                # Only continue if this is a better path
                if (
                    neighbor not in reached_states
                    or g_new < reached_states[neighbor]
                ):

                    reached_states[neighbor] = g_new

                    counter += 1

                    new_path = path_taken + [action]

                    heapq.heappush(
                        frontier,
                        (
                            f_new,
                            g_new,
                            counter,
                            neighbor,
                            new_path
                        )
                    )

        return []

    # ==========================================================
    # FIND CLOSEST FOOD
    # ==========================================================

    def find_closest_food(
        self,
        start,
        food_positions,
        grid_size,
        walls
    ):
        """
        Find the closest reachable food using BFS.

        Returns:
            Position of closest food.
        """

        closest_food = None
        shortest_path = None

        for food in food_positions:

            path = self.bfs_search(
                start,
                food,
                grid_size,
                walls
            )

            # Ignore unreachable food
            if not path and start != food:
                continue

            if (
                shortest_path is None
                or len(path) < len(shortest_path)
            ):

                shortest_path = path
                closest_food = food

        return closest_food

    # ==========================================================
    # SEARCH SELECTOR
    # ==========================================================

    def search(
        self,
        start,
        goal,
        grid_size,
        walls
    ):
        """
        Select the search algorithm according to
        self.active_algo.
        """

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

        elif self.active_algo == "AStar":

            return self.astar_search(
                start_pos=start,
                goal_pos=goal,
                walls=walls,
                grid_size=grid_size,
                heuristic_type="manhattan"
            )

        else:

            print(
                f"Invalid algorithm: {self.active_algo}"
            )

            return []

    # ==========================================================
    # SENSE AND ACT
    # ==========================================================

    def sense_and_act(self, percept):
        """
        Receive percept and return the next action.
        """

        # ------------------------------------------------------
        # Create a new plan when no plan exists
        # ------------------------------------------------------

        if not self.plan:

            start = percept["position"]

            grid_size = percept["grid_size"]

            walls = set(
                percept["walls"]
            )

            food_positions = set(
                percept["all_food"]
            )

            # --------------------------------------------------
            # If there is no food left
            # --------------------------------------------------

            if not food_positions:

                return "Stay"

            # --------------------------------------------------
            # Find closest food
            # --------------------------------------------------

            goal = self.find_closest_food(
                start,
                food_positions,
                grid_size,
                walls
            )

            # --------------------------------------------------
            # If no reachable food exists
            # --------------------------------------------------

            if goal is None:

                return "Stay"

            # --------------------------------------------------
            # A* algorithm
            # --------------------------------------------------

            if self.active_algo == "AStar":

                self.plan = self.astar_search(
                    start_pos=start,
                    goal_pos=goal,
                    walls=walls,
                    grid_size=grid_size,
                    heuristic_type="manhattan"
                )

            # --------------------------------------------------
            # Other algorithms
            # --------------------------------------------------

            else:

                self.plan = self.search(
                    start,
                    goal,
                    grid_size,
                    walls
                )

        # ------------------------------------------------------
        # Execute next action from plan
        # ------------------------------------------------------

        if self.plan:

            return self.plan.pop(0)

        return "Stay"