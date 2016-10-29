from chainer import computational_graph as c


class ComputationalGraph:
    def __init__(self, path_to_graph):
        self.path = path_to_graph

    def __call__(self, vl):
        graph = c.build_computational_graph(vl)
        with open(self.path, 'w') as f:
            f.write(graph.dump())
