class PathPLan:
    def __init__(self):
        # landmark coords
        # box (obstacle) coords
        # get offsetted coordinates for shortest path (box coords + radius)
                # if box cordinate is on bottom left, offsetted coords would be (x, y + offset) and (x - offset, y)
        # get offsetted coordinates for voronoi coords = create_voronoi_roadmap()
        # self.adj_list = create_adjacency_list()
        # get radius of car (constant)
        # start point
        # goal point

    def _in_obstacle(self, point):
        # check if point is in obstacle

    def create_adjacency_list(self):
        # land mark bounds of world (offset by radius of car)
        # obstacles bounds (offset by radius of car)
        # connect 2 vertices if a straight line between 2 points doesnt go through the obstacle --> use _in_obstacle
    
    def a_star(self):
        # do a* using heuristic as distance from point to goal (straight line even if it cuts through obstacle)

    def create_voronoi_roadmap(self):
        # get half way points between each edge of box and respective edge in world (along the)

    def get_shortest_path(self):
        # return respective waypoints

    def get_max_safety(self):
        # return respective waypoints
    

