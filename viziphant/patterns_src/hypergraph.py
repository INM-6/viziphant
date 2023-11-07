import numpy as np
import networkx as nx


class Hypergraph:
    """
    Hypergraph data structure
    An object of this class represents a weighted hypergraph
    and provides some utility functions.
    A hypergraph consists of a set of vertices and a set of hyperedges.
    Every hyperedge is a set of vertices.
    Additional information such as labels and positions of vertices
    and weights of hyperedges can be defined as well.
    """

    def __init__(self, hyperedges, vertices, vertex_labels=None, weights=None,
                 positions=None, repulse=True):
        """
        Constructor of a Hypergraph object.

        Parameters
        ----------
        hyperedges: list of lists
            List of hyperedges. Every hyperedge is a list of vertices, i.e.,
            values listed in parameter `vertices`.
            Only unique hyperedges will be saved.
        vertices: list
            List of vertices of the hypergraph. Items may be arbitrary
            hashable values.
        vertex_labels: list (optional)
            Labels of the vertices,
            vertex_labels[i] is the label of vertices[i].
        weights: list of int or float (optional)
            Weights of the hyperedges.
            Every item corresponds to one hyperedge:
            weights[i] is the weight of hyperedge hyperedges[i].
            Thus, len(weights) == len(hyperedges) is required.
        positions: np.ndarray (optional)
            Array of shape (len(vertices), 2)
            containing positions of the vertices
        repulse: bool (optional)
            Whether vertices should repel each other more strongly if they are
            in overlapping hyperedges but not in the same hyperedge
        """

        self.hyperedges = hyperedges
        self.weights = weights
        self.vertices = sorted(list(vertices))
        self.vertex_labels = vertex_labels
        self.positions = positions
        self.repulse = repulse

        # Make hyperedges unique
        # Sort every hyperedge and make it hashable by making it a tuple
        # instead of a list
        hyperedges_and_weights = [tuple(sorted(x)) for x in self.hyperedges]
        # Constructing a set out of the list of hyperedges makes entries unique
        hyperedges_and_weights = set(zip(hyperedges_and_weights, self.weights))
        # Sort hyperedges lexicographically (by first non-equal element)
        hyperedges_and_weights = list(sorted(hyperedges_and_weights))
        # Make every hyperedge a list again
        self.hyperedges = [list(x[0]) for x in hyperedges_and_weights]
        self.weights = [x[1] for x in hyperedges_and_weights]

        # Hyperedges need to be saved with respect to vertices
        # from a 0-based range in order to use them for indexing
        # into the positions array or similar
        self.hyperedges_indexes = []
        # Mapping of every vertex of all hyperedges to their indices in an
        # ordered list of unique vertices
        for h in self.hyperedges:
            # New hyperedge
            self.hyperedges_indexes.append([])
            # Append index of every vertex to the current hyperedge
            for v in h:
                self.hyperedges_indexes[-1].append(self.vertices.index(v))

    # TODO: Treat weights of overlapping hyperedges, some binary edges occur
    # multiple times

    def complete_and_star_associated_graph(self):
        """
        Transformation of the hypergraph into a union of its
        complete associated graph and its star associated graph
        The resulting graph contains all vertices of the hypergraph
        and one pseudo-vertex for every hyperedge.
        Every pair of adjacent vertices in the hypergraph
        (i.e., vertices that are in at least one common hyperedge)
        becomes an edge of the resulting graph.
        An edge is also constructed for every vertex of a hyperedge
        joining it with the pseudo-vertex corresponding to the hyperedge.

        The complete and star associated graph are defined in:
        Arafat and Bressan: Hypergraph drawing by Force-Directed Placement
        https://doi.org/10.1007/978-3-319-64471-4_31

        Returns
        -------
        G: nx.Graph
            A networkx graph object containing the complete associated graph
            of this hypergraph
        """
        # TODO: positions
        edges = []
        weights = []
        graph_vertices = list(self.vertices.copy())
        for i, hyperedge in enumerate(self.hyperedges):
            # Pseudo-vertex corresponding to hyperedge
            graph_vertices.append(-i - 1)
            for j, vertex in enumerate(hyperedge):
                # Every vertex of a hyperedge is adjacent to the pseudo-vertex
                # corresponding to the hyperedge
                edges.append([-i - 1, vertex])
                # Weight is equal to the weight of the hyperedge (if
                # applicable)
                if self.weights:
                    weights.append(self.weights[i])
                # Unique unordered combinations of vertices of this hyperedge
                for k in range(j + 1, len(hyperedge)):
                    # Save the combination as an edge
                    edges.append([vertex, hyperedge[k]])
                    # If the hypergraph is weighted, the resulting edge has the
                    # same weight as the original hyperedge
                    if self.weights:
                        weights.append(self.weights[i])

        # Create a mapping of edges to their weights as tuples => input to
        # networkx
        if weights:
            edges_nx = [(edges[i][0], edges[i][1], {'weight': weights[i]})
                        for i in range(len(edges))]
        else:
            edges_nx = edges
        # Construct a graph from this
        G = nx.Graph()
        # All vertices of the hypergraph
        G.add_nodes_from(graph_vertices)
        # Edges as constructed above
        G.add_edges_from(edges_nx)
        return G

    def complete_associated_graph(self):
        """
        Transformation of the hypergraph into its complete associated graph
        The resulting graph contains all vertices of the hypergraph.
        Every pair of adjacent vertices in the hypergraph
        (i.e., vertices that are in at least one common hyperedge)
        becomes an edge of the resulting graph.
        The weight of an edge is the weight of the hyperedge that joins its
        pair of vertices.
        If the pair of vertices is in multiple hyperedges, the weight of the
        resulting edge is an arbitrary one of the corresponding hyperedge
        weights.

        The complete associated graph is defined in:
        Arafat and Bressan: Hypergraph drawing by Force-Directed Placement
        https://doi.org/10.1007/978-3-319-64471-4_31

        Returns
        -------
        G: nx.Graph
            A networkx graph object containing the complete associated graph
            of this hypergraph
        """

        # TODO: positions
        edges = []
        weights = []
        for i, hyperedge in enumerate(self.hyperedges):
            # Every unique unordered combination of vertices of this hyperedge
            for j, vertex in enumerate(hyperedge):
                for k in range(j + 1, len(hyperedge)):
                    # Save the combination as an edge
                    edges.append([vertex, hyperedge[k]])
                    # If the hypergraph is weighted, the resulting edge has the
                    # same weight as the original hyperedge
                    if self.weights:
                        weights.append(self.weights[i])

        if self.repulse:
            # Make vertices repulse if they are in overlapping hyperedges
            # but not in the same
            # For every pair of hyperedges
            for i, he1 in enumerate(self.hyperedges):
                for j, he2 in enumerate(self.hyperedges):
                    # If they do not overlap or are the same, nothing needs to
                    # be done
                    overlap = set(he1).intersection(he2)
                    if i == j or len(overlap) == 0:
                        continue
                    # Every pair of vertices in these hyperedges
                    for v1 in he1:
                        for v2 in he2:
                            # If one of the vertices is in the intersection,
                            # it shares a common hyperedge with the other
                            # vertex
                            if v1 not in overlap and v2 not in overlap:
                                # Add this edge with a low weight
                                edges.append([v1, v2])
                                # Low weight means repulsion
                                if self.weights:
                                    weights.append(0.1)

        # Create a mapping of edges to their weights as tuples => input to
        # networkx
        if weights:
            edges_nx = [(edges[i][0], edges[i][1], {'weight': weights[i]})
                        for i in range(len(edges))]
        else:
            edges_nx = edges
        # Construct a graph from this
        G = nx.Graph()
        # All vertices of the hypergraph
        G.add_nodes_from(self.vertices)
        # Edges as constructed above
        G.add_edges_from(edges_nx)
        return G

    def complete_weighted_associated_graph(self):
        """
        Transformation of the hypergraph into a graph suitable for graph layout
        algorithms
        The resulting graph is based on the complete associated graph
        of the hypergraph.
        All edges of the complete associated graph are kept,
        including their weights.
        All other pairs of vertices are added as edges with a weight of 1
        such that the resulting graph is a complete graph with the weights
        distinguishing between edges of the complete associated graph
        of the hypergraph and the newly introduced edges.

        The complete associated graph is defined in:
        Arafat and Bressan: Hypergraph drawing by Force-Directed Placement
        https://doi.org/10.1007/978-3-319-64471-4_31


        Returns
        -------
        G: nx.Graph
            A networkx graph object containing a complete weighted graph
            representing the complete associated graph of this hypergraph
        """

        import itertools as it
        # All pairs of distinct vertices become are edges of the complete graph
        all_edges = np.array(list(
                it.combinations_with_replacement(self.vertices, 2)))
        G = nx.Graph()
        # All edges are weighted with weight 1 in the beginning
        G.add_edges_from(all_edges, weight=1)
        # Replace edges contained in the complete associated graph
        # with their originals
        # => Weights are changed to the original weights
        G.update(self.complete_associated_graph())
        return G

    @staticmethod
    def get_union_of_vertices(hypergraphs):
        """
        Get a list of unique vertices of all given hypergraphs and
        corresponding vertex labels.
        In case a vertex occurs in multiple hypergraphs it will be listed
        only once.
        If the labels for this vertex are not the same in all hypergraphs,
        an arbitrary one of those is chosen.

        Parameters
        ----------
        hypergraphs: list of Hypergraph
            List of hypergraphs. The vertices and corresponding labels will
            be extracted from these hypergraphs.

        Returns
        -------
        vertices: list
           List of all unique vertices of the hypergraphs
        vertex_labels: list
            List of labels corresponding to the vertices.
            vertex_labels[i] is the label for vertices[i].
        """

        # Find the set of all common vertices while preserving mapping
        # from vertex (arbitrary ID) to vertex label
        # Dict: vertex -> vertex label
        vertices_to_vertex_labels = {}
        # Add all vertices from all hypergraphs
        for hg in hypergraphs:
            for i in range(len(hg.vertices)):
                # TODO: Conflicts of labels are currently ignored, the last
                # label is currently used in this case.
                if hg.vertex_labels:
                    vertices_to_vertex_labels[hg.vertices[i]] = \
                        hg.vertex_labels[i]
                else:
                    vertices_to_vertex_labels[hg.vertices[i]] = None
        # Unique pairs of (vertex, vertex label)
        vertices_to_vertex_labels = vertices_to_vertex_labels.items()
        # If there are no vertices, empty lists are returned
        if not vertices_to_vertex_labels:
            return [], []
        # Split them into separate lists: [(v1, l1), ..., (vn, ln)] -> ([v1,
        # ..., vn], [l1, ..., ln])
        vertices, vertex_labels = tuple(zip(*vertices_to_vertex_labels))
        # Vertices as numpy array to enable array indexing
        vertices = np.array(vertices)

        return vertices, vertex_labels

    def layout_hypergraph(self):
        """
        Compute a layout for this hypergraph such that adjacent vertices
        are close together
        and non-adjacent vertices are further apart.
        The hypergraph is transformed into the complete and weighted
        representation of the complete associated graph
        and the Fruchterman-Reingold layout algorithm is applied to it.
        The resulting positions of vertices can be used to visualize
        the hypergraph in such a way that drawing hyperedges produces
        aesthetically pleasing results due to minimizing undesired overlap.

        Returns
        -------
        pos: np.ndarray
            Array of shape (len(self.vertices), 2)
            containing computed positions of the vertices
        """

        # Apply Fruchterman-Reingold graph layout algorithm to the
        # complete weighted representation of the
        # complete associated graph of this hypergraph
        # Layout algorithm returns a dictionary {vertexID -> position}
        pos_dict = nx.fruchterman_reingold_layout(
            self.complete_weighted_associated_graph(),
            scale=5, k=1/5, center=[5,5], iterations=1000)
        # Turn dictionary into a contiguous array of positions
        pos = list(map(lambda n: pos_dict[n], self.vertices))
        pos = np.array(pos)
        return pos
