from collections import deque
import heapq
import math
import random

from logic_engine import KnowledgeBase


# ==============================================================
# PRACTICAL 01 - SIMPLE REFLEX AGENT
# ==============================================================

class SimpleReflexAgent:
    """
    Simple Reflex Agent.

    Chooses an action using only the current percept.
    It does not maintain memory of previous states.
    """

    def __init__(self):
        self.actions_pool = ["Up", "Down", "Left", "Right"]

    def sense_and_act(self, percept):
        """
        React only to the current percept.
        """

        wall_ahead = percept.get("wall_ahead", False)
        food_here = percept.get("food_here", False)

        # If food is currently available, remain on the tile.
        if food_here:
            return "Stay"

        # If a wall is detected, choose another valid direction.
        if wall_ahead:
            return random.choice(
                ["Left", "Right", "Down", "Up"]
            )

        return random.choice(self.actions_pool)


# ==============================================================
# PRACTICAL 02 - MODEL-BASED AGENT
# ==============================================================

class ModelBasedAgent:
    """
    Model-Based Agent.

    Maintains internal state so that it can remember previous
    actions and avoid repeatedly making the same decision.
    """

    def __init__(self):
        self.actions_pool = ["Up", "Down", "Left", "Right"]

        # Internal memory
        self.last_action = None

    def sense_and_act(self, percept):
        """
        Choose an action using the current percept and memory.
        """

        wall_ahead = percept.get("wall_ahead", False)
        food_here = percept.get("food_here", False)

        if food_here:
            return "Stay"

        possible_actions = self.actions_pool.copy()

        # If a wall is detected, avoid repeating the previous
        # movement action.
        if wall_ahead:

            if self.last_action in possible_actions:
                possible_actions.remove(self.last_action)

            action = random.choice(possible_actions)

            self.last_action = action

            return action

        # Also avoid repeating the previous action where possible.
        if self.last_action in possible_actions:
            possible_actions.remove(self.last_action)

        action = random.choice(possible_actions)

        self.last_action = action

        return action


# ==============================================================
# PRACTICALS 03, 04 AND 05 - SEARCH AGENT
# ==============================================================

class SearchAgent:
    """
    Search Agent for IT3012.

    Practical 03:
        - BFS
        - DFS
        - UCS

    Practical 04:
        - A*
        - Manhattan heuristic
        - Euclidean heuristic

    Practical 05:
        - Knowledge Base
        - Horn Clause rules
        - Forward Chaining
        - Logical feasibility checking
    """

    def __init__(self):

        self.plan = []

        self.active_algo = "BFS"

        # ======================================================
        # PRACTICAL 05 - KNOWLEDGE BASE
        # ======================================================

        self.kb = KnowledgeBase()

        # Rule 1:
        # TargetVisible AND HasDust -> SafeToEngage
        self.kb.tell_rule(
            ["TargetVisible", "HasDust"],
            "SafeToEngage"
        )

        # Rule 2:
        # SafeToEngage AND BloodseekerMissing -> Retreat
        self.kb.tell_rule(
            ["SafeToEngage", "BloodseekerMissing"],
            "Retreat"
        )

    # ==========================================================
    # GET NEIGHBORS
    # ==========================================================

    def get_neighbors(self, state, grid_size, walls):

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

            # Grid boundary check
            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Physical reachability check
            if position in walls:
                continue

            neighbors.append(
                (position, action)
            )

        return neighbors

    # ==========================================================
    # BFS
    # ==========================================================

    def bfs_search(
        self,
        start,
        goal,
        walls,
        grid_size
    ):

        queue = deque()

        queue.append(
            (start, [])
        )

        reached = {start}

        while queue:

            current, path = queue.popleft()

            if current == goal:
                return path

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
    # DFS
    # ==========================================================

    def dfs_search(
        self,
        start,
        goal,
        walls,
        grid_size
    ):

        stack = []

        stack.append(
            (start, [])
        )

        reached = {start}

        while stack:

            current, path = stack.pop()

            if current == goal:
                return path

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
    # UCS
    # ==========================================================

    def ucs_search(
        self,
        start,
        goal,
        walls,
        grid_size
    ):

        frontier = []

        counter = 0

        heapq.heappush(
            frontier,
            (
                0,
                counter,
                start,
                []
            )
        )

        reached = {
            start: 0
        }

        while frontier:

            cost, _, current, path = heapq.heappop(
                frontier
            )

            if current == goal:
                return path

            for neighbor, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):

                new_cost = cost + 1

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

        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    # ==========================================================
    # EUCLIDEAN DISTANCE
    # ==========================================================

    def euclidean_distance(self, pos, goal):

        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )

    # ==========================================================
    # PRACTICAL 05 - LOGICAL FEASIBILITY
    # ==========================================================

    def is_feasible(
        self,
        position,
        tile_percepts
    ):
        """
        Check whether a physically reachable tile is
        logically feasible.

        If the KB deduces 'Retreat', the position is
        considered infeasible.
        """

        # Clear facts from previous candidate tile
        self.kb.clear_facts()

        # Obtain percepts for this tile
        percepts = tile_percepts.get(
            position,
            []
        )

        # Add percepts as KB facts
        for fact in percepts:
            self.kb.tell_fact(fact)

        # Perform inference
        self.kb.forward_chain()

        # Retreat means logically infeasible
        if "Retreat" in self.kb.facts:
            return False

        return True

    # ==========================================================
    # A* SEARCH
    # ==========================================================

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type="manhattan",
        tile_percepts=None
    ):

        if tile_percepts is None:
            tile_percepts = {}

        # Select heuristic
        if heuristic_type.lower() == "euclidean":
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        frontier = []

        g_cost = 0

        h_cost = heuristic(
            start_pos,
            goal_pos
        )

        f_cost = g_cost + h_cost

        counter = 0

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

            # Ignore outdated frontier entries
            if (
                current_pos in reached_states
                and current_g > reached_states[current_pos]
            ):
                continue

            # Goal test
            if current_pos == goal_pos:
                return path_taken

            # Expand current node
            for neighbor, action in self.get_neighbors(
                current_pos,
                grid_size,
                walls
            ):

                # ==============================================
                # PRACTICAL 05 FEASIBILITY CHECK
                # ==============================================

                if not self.is_feasible(
                    neighbor,
                    tile_percepts
                ):
                    continue

                # ==============================================
                # NORMAL A* PROCESSING
                # ==============================================

                g_new = current_g + 1

                h_new = heuristic(
                    neighbor,
                    goal_pos
                )

                f_new = g_new + h_new

                if (
                    neighbor not in reached_states
                    or g_new < reached_states[neighbor]
                ):

                    reached_states[neighbor] = g_new

                    counter += 1

                    new_path = (
                        path_taken + [action]
                    )

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

        closest_food = None
        shortest_path = None

        for food in food_positions:

            path = self.bfs_search(
                start,
                food,
                walls,
                grid_size
            )

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
        walls,
        tile_percepts=None
    ):

        if self.active_algo == "BFS":

            return self.bfs_search(
                start,
                goal,
                walls,
                grid_size
            )

        elif self.active_algo == "DFS":

            return self.dfs_search(
                start,
                goal,
                walls,
                grid_size
            )

        elif self.active_algo == "UCS":

            return self.ucs_search(
                start,
                goal,
                walls,
                grid_size
            )

        elif self.active_algo == "AStar":

            return self.astar_search(
                start_pos=start,
                goal_pos=goal,
                walls=walls,
                grid_size=grid_size,
                heuristic_type="manhattan",
                tile_percepts=tile_percepts
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

        if not self.plan:

            start = percept["position"]

            grid_size = percept["grid_size"]

            walls = set(
                percept["walls"]
            )

            food_positions = set(
                percept["all_food"]
            )

            # Practical 05 percept information
            tile_percepts = percept.get(
                "tile_percepts",
                {}
            )

            # No food remaining
            if not food_positions:
                return "Stay"

            # Find closest food
            goal = self.find_closest_food(
                start,
                food_positions,
                grid_size,
                walls
            )

            if goal is None:
                return "Stay"

            # A*
            if self.active_algo == "AStar":

                self.plan = self.astar_search(
                    start_pos=start,
                    goal_pos=goal,
                    walls=walls,
                    grid_size=grid_size,
                    heuristic_type="manhattan",
                    tile_percepts=tile_percepts
                )

            # BFS / DFS / UCS
            else:

                self.plan = self.search(
                    start,
                    goal,
                    grid_size,
                    walls,
                    tile_percepts
                )

        if self.plan:
            return self.plan.pop(0)

        return "Stay"