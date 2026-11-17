import math
import heapq
from itertools import combinations
import yaml


config_path = './world_map.yaml'
class PathPlan:
    def __init__(self):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.landmarks = config["landmarks"]
        self.start = tuple(config["start"])
        self.goal = tuple(config["goal"])

        self.radius = config["robot"]["radius"]
        self.clearance = self.radius + config["robot"]["tolerance"]

        self.world = self.get_bounds(config["world"]["vertices"])
        self.box = self.get_bounds(config["obstacle"]["vertices"])

        self.shortest_vertices = self.create_shortest_roadmap()
        self.voronoi_vertices = self.create_voronoi_roadmap()

    def get_bounds(self, vertices):
        x, y = zip(*vertices)
        return min(x), min(y), max(x), max(y)
    

    def _in_obstacle(self, point):
        x, y = point
        xmin, ymin, xmax, ymax = self.box
        r = self.clearance

        return (
            xmin - r <= x <= xmax + r
            and ymin - r <= y <= ymax + r
        )

    def _in_world(self, point):
        x, y = point
        xmin, ymin, xmax, ymax = self.world
        r = self.clearance

        return (
            xmin + r <= x <= xmax - r
            and ymin + r <= y <= ymax - r
        )

    def _collision_free(self, p1, p2):
        if not self._in_world(p1) or not self._in_world(p2):
            return False

        if self._in_obstacle(p1) or self._in_obstacle(p2):
            return False

        xmin, ymin, xmax, ymax = self.box
        r = self.clearance

        # Check intersection with the expanded obstacle
        t_min, t_max = 0.0, 1.0

        for start, end, lower, upper in [
            (p1[0], p2[0], xmin - r, xmax + r),
            (p1[1], p2[1], ymin - r, ymax + r)
        ]:
            delta = end - start

            if delta == 0:
                if not lower <= start <= upper:
                    return True
                continue

            t1 = (lower - start) / delta
            t2 = (upper - start) / delta

            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

            if t_min > t_max:
                return True

        return False

    def create_shortest_roadmap(self):
        xmin, ymin, xmax, ymax = self.box

        # Extra offset prevents diagonal edges from
        # cutting through the expanded obstacle corners.
        d = 2 * self.clearance + 0.001

        return [
            (xmin, ymin - d),   # A
            (xmax, ymin - d),   # B
            (xmax + d, ymin),   # C
            (xmax + d, ymax),   # D
            (xmax, ymax + d),   # E
            (xmin, ymax + d),   # F
            (xmin - d, ymax),   # G
            (xmin - d, ymin)    # H
        ]

    def create_voronoi_roadmap(self):
        wxmin, wymin, wxmax, wymax = self.world
        xmin, ymin, xmax, ymax = self.box

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        return [
            (center_x, (wymin + ymin) / 2),  # Bottom
            ((wxmax + xmax) / 2, center_y),  # Right
            (center_x, (wymax + ymax) / 2),   # Top
            ((wxmin + xmin) / 2, center_y)   # Left
        ]

    def create_adjacency_list(self, vertices):
        vertices = list(dict.fromkeys(
            [self.start] + vertices + [self.goal]
        ))

        adj_list = {point: [] for point in vertices}

        for p1, p2 in combinations(vertices, 2):

            if self._collision_free(p1, p2):

                weight = math.dist(p1, p2)

                adj_list[p1].append((p2, weight))
                adj_list[p2].append((p1, weight))

        return adj_list

    def a_star(self, adj_list):
        queue = [(math.dist(self.start, self.goal), self.start)]

        distance = {self.start: 0}
        parent: dict[tuple[float, float], tuple[float, float] | None] = {
            self.start: None
        }

        while queue:
            priority, current = heapq.heappop(queue)

            if priority > distance[current] + math.dist(current, self.goal):
                continue

            if current == self.goal:
                path = []

                while current is not None:
                    path.append(current)
                    current = parent[current]

                path.reverse()

                # Calculate heading toward the next waypoint
                waypoints = []

                for i in range(len(path) - 1):
                    x, y = path[i]
                    next_x, next_y = path[i + 1]

                    theta = math.atan2(next_y - y, next_x - x)
                    waypoints.append((x, y, theta))

                # Final waypoint retains the previous heading
                x, y = path[-1]
                theta = waypoints[-1][2] if waypoints else 0.0
                waypoints.append((x, y, theta))

                return waypoints

            for neighbor, weight in adj_list[current]:

                new_distance = distance[current] + weight

                if new_distance < distance.get(neighbor, math.inf):

                    distance[neighbor] = new_distance
                    parent[neighbor] = current

                    heuristic = math.dist(neighbor, self.goal)

                    heapq.heappush(
                        queue,
                        (new_distance + heuristic, neighbor)
                    )

        return []

    def get_shortest_path(self):
        adj_list = self.create_adjacency_list(
            self.shortest_vertices
        )

        return self.a_star(adj_list)

    def get_max_safety(self):
        adj_list = self.create_adjacency_list(
            self.voronoi_vertices
        )

        return self.a_star(adj_list)