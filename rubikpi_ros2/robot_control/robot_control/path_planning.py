import math
import heapq
from itertools import combinations
import yaml
import numpy as np

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
        x,y = zip(*vertices)
        
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        return x_min, x_max, y_min, y_max
    

    def _in_obstacle(self, point):
        x, y = point
        x_min, x_max, y_min, y_max = self.box
        r = self.clearance

        return x_min + r <= x <= x_max -r and y_min + r <= y <= y_max - r

    def _in_world(self, point):
        x, y = point
        x_min, x_max, y_min, y_max = self.world
        r = self.clearance

        return (
            x_min + r <= x <= x_max - r
            and y_min + r <= y <= y_max - r
        )

    def _collision_free(self, p1, p2):
        if not self._in_world(p1) or not self._in_world(p2):
            return False

        if self._in_obstacle(p1) or self._in_obstacle(p2):
            return False

        xmin, xmax, ymin, ymax = self.box
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
        xmin, xmax, ymin, ymax = self.box
     
        # Extra offset prevents diagonal edges from
        # cutting through the expanded obstacle corners.
        d = self.clearance 

        point_distance_from_corner = d*np.sin(math.pi/4)
        return [
            (xmin - point_distance_from_corner, ymin - point_distance_from_corner),   # A
            (xmax + point_distance_from_corner, ymin - point_distance_from_corner),   # D
            (xmax + point_distance_from_corner, ymax + point_distance_from_corner),   # E
            (xmin - point_distance_from_corner, ymax + point_distance_from_corner),   # F
        ]

    def create_voronoi_roadmap(self):
        wxmin, wxmax, wymin, wymax = self.world
        xmin, xmax, ymin, ymax= self.box

        return [
            ((xmin + wxmin)/2, (wymin + ymin) / 2),  # Bottom
            ((wxmax + xmax) / 2, (ymin + wymin)/2),  # Right
            ((xmax + wxmax)/2, (wymax + ymax) / 2),   # Top
            ((wxmin + xmin) / 2, (ymax + wymax)/2)   # Left
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
# def main():
#     pathplan = PathPlan()
#     print("World coords ", pathplan.world)
#     print("Box coords ", pathplan.box)
#     print("Car radius ", pathplan.radius)
#     print("Car offset ", pathplan.clearance)
#     print("SP ", pathplan.shortest_vertices)
#     print("MS ", pathplan.voronoi_vertices)
#     print("Start ", pathplan.start)
#     print("Goal ", pathplan.goal)
# main()