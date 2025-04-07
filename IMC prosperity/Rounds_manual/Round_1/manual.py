class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.vertex_data = [''] * size

    def add_edge(self, u, v, weight):
        if 0 <= u < self.size and 0 <= v < self.size:
            self.adj_matrix[u][v] = weight
            self.adj_matrix[v][u] = weight  # For undirected graph

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data

def dijkstra(self, start_vertex_data):
    start_vertex = self.vertex_data.index(start_vertex_data)
    distances = [float('inf')] * self.size
    distances[start_vertex] = 0
    visited = [False] * self.size

    for _ in range(self.size):
        min_distance = float('inf')
        u = None
        for i in range(self.size):
            if not visited[i] and distances[i] < min_distance:
                min_distance = distances[i]
                u = i

        if u is None:
            break

        visited[u] = True

        for v in range(self.size):
            if self.adj_matrix[u][v] != 0 and not visited[v]:
                alt = distances[u] + self.adj_matrix[u][v]
                if alt < distances[v]:
                    distances[v] = alt

    return distances

trades = Graph(4)
trades.add_edge("snowball","pizzas",1.45)
trades.add_edge("pizzas","snowball",0.7)
trades.add_edge("snowball","nuggets",0.52)
trades.add_edge("nuggets","snowball",1.95)
trades.add_edge("snowball","shells",0.72)
trades.add_edge("shells","snowball",1.34)
trades.add_edge("snowbpizzasll","shells",0.72)
trades.add_edge("shells","pizzas",1.34)