"""
Unit tests for the Hypergraph class.
"""

import unittest
import numpy as np

# import quantities as pq
from numpy.testing import assert_array_equal
from viziphant.patterns_src.hypergraph import Hypergraph
import networkx as nx


class HypergraphTestCase(unittest.TestCase):
    """
    This test is constructed to test the Creation and Transformation of the Hypergraph Class.
    In the Setup function different Hypergraphs are created and being tested on different
    Graph Transformation algorithms.
    """

    def setUp(self):
        self.hypergraphs = []
        # self.spiketrains = elephant.spike_train_generation.compound_poisson_process(
        #    rate=5 * pq.Hz,
        #    amplitude_distribution=[0] + [0.98] + [0] * 8 + [0.02],
        #    t_stop=10 * pq.s,
        # )
        # self.patterns = elephant.spade.spade(
        #    spiketrains=self.spiketrains,
        #    binsize=1 * pq.ms,
        #    winlen=1,
        #    min_spikes=2,
        #    n_surr=100,
        #    dither=5 * pq.ms,
        #    psr_param=[0, 0, 0],
        #    output_format="patterns",
        # )["patterns"]

        # Hypergraph with characters as vertices
        self.hyperedges = [["a", "b"], ["b", "c"], ["c", "d"]]
        self.vertices = ["a", "b", "c", "d"]
        self.vertex_labels = ["A", "B", "C", "D"]
        self.weights = [1, 2, 3]
        self.positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.repulse = True

        self.hypergraphs.append(
            Hypergraph(
                self.hyperedges,
                self.vertices,
                self.vertex_labels,
                self.weights,
                self.positions,
                self.repulse,
            )
        )

        # normal Hypergraph
        self.hyperedges = [[1, 2], [2, 3], [3, 4, 9], [5, 6, 7, 8]]
        self.vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.vertex_labels = [
            "neuron1",
            "neuron2",
            "neuron3",
            "neuron4",
            "neuron5",
            "neuron6",
            "neuron7",
            "neuron8",
            "neuron9",
        ]
        self.weights = [1, 2, 1, 1]
        self.positions = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7],
                [8, 8],
                [9, 9],
            ]
        )
        self.repulse = True

        self.hypergraphs.append(
            Hypergraph(
                self.hyperedges,
                self.vertices,
                self.vertex_labels,
                self.weights,
                self.positions,
                self.repulse,
            )
        )

        # Hypergraph with negative Vertices
        self.hyperedges = [[-1, -2], [-2, 3], [3, 4]]
        self.vertices = [-1, -2, 3, 4]
        self.vertex_labels = ["neuron1", "neuron2", "neuron3", "neuron4"]
        self.weights = [3, 4, 5]
        self.positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.repulse = False

        self.hypergraphs.append(
            Hypergraph(
                self.hyperedges,
                self.vertices,
                self.vertex_labels,
                self.weights,
                self.positions,
                self.repulse,
            )
        )

        # Hypergraph without weights and labels
        self.hyperedges = [["a", "b"], ["b", "c"], ["c", "d"]]
        self.vertices = ["a", "b", "c", "d"]
        self.vertex_labels = ["","","",""]
        self.weights = []
        self.positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.repulse = False

        self.hypergraphs.append(
            Hypergraph(
                self.hyperedges,
                self.vertices,
                self.vertex_labels,
                self.weights,
                self.positions,
                self.repulse,
            )
        )

    def test_hypergraph_init(self):
        hyperedges = [["a", "b"], ["b", "c"], ["c", "d"]]
        vertices = ["a", "b", "c", "d"]
        vertex_labels = ["A", "B", "C", "D"]
        weights = [1, 2, 3]
        positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges, vertices, vertex_labels, weights, positions, repulse,
        )
        # check the Types
        self.assertTrue(hypergraph.hyperedges == hyperedges)
        self.assertTrue(hypergraph.vertices == vertices)
        self.assertTrue(hypergraph.vertex_labels == vertex_labels)
        self.assertTrue(hypergraph.weights == weights)
        self.assertTrue(np.array_equal(hypergraph.vertices, vertices))
        self.assertTrue(hypergraph.repulse)
        self.assertEqual(len(hypergraph.weights), len(hypergraph.hyperedges))

    # Test of getter method
    def test_get_union_of_vertices(self):
        for hypergraph in self.hypergraphs:
            new_vertices = hypergraph.get_union_of_vertices([hypergraph])[0]
            new_vertex_labels = hypergraph.get_union_of_vertices([hypergraph])[1]
            # check equality after creation
            assert_array_equal(new_vertices, hypergraph.vertices)
            assert_array_equal(new_vertex_labels, hypergraph.vertex_labels)

    def test_layout_hypergraph_len(self):
        # nx.fruchterman_reingold_layout algorithm applied
        for hypergraph in self.hypergraphs:
            # check if data is consistent
            self.assertEqual(
                len(hypergraph.layout_hypergraph()), len(hypergraph.vertices)
            )

    # Test complete and star associated graph layouting algorithm
    def test_complete_and_star_associated_graph(self):
        for hypergraph in self.hypergraphs:
            graph = hypergraph.complete_and_star_associated_graph()
            self.assertIsInstance(graph, nx.Graph)
            for vertex in hypergraph.vertices:
                self.assertIn(vertex, list(graph.nodes))

            edges = []
            for elem in list(graph.edges):
                edges.extend(elem)

            for edge in hypergraph.hyperedges:
                for node in edge:
                    self.assertIn(node, edges)

            # every hyperedge needs a weight
            self.assertEqual(len(hypergraph.hyperedges), len(hypergraph.weights))

            # nodes in graph equal nodes + len(hyperedges) because of pseudo vertices
            self.assertEqual(
                len(graph.nodes), len(hypergraph.vertices) + len(hypergraph.hyperedges)
            )
            # self.assertEqual(len(graph.edges), 9)

    # Test complete and associated graph layouting algorithm
    def test_complete_associated_graph(self):
        for hypergraph in self.hypergraphs:
            graph = hypergraph.complete_associated_graph()
            # algorithm returns nx.Graph
            self.assertIsInstance(graph, nx.Graph)
            for vertex in hypergraph.vertices:
                self.assertIn(vertex, list(graph.nodes))

            edges = []
            for elem in list(graph.edges):
                edges.extend(elem)

            for edge in hypergraph.hyperedges:
                for node in edge:
                    self.assertIn(node, edges)

            self.assertEqual(len(hypergraph.hyperedges), len(hypergraph.weights))
            assert_array_equal(list(graph.nodes), hypergraph.vertices)

    # Test complete and weight associated graph layouting algorithm
    def test_complete_weighted_associated_graph(self):
        for hypergraph in self.hypergraphs:
            graph = hypergraph.complete_weighted_associated_graph()
            self.assertIsInstance(graph, nx.Graph)
            for vertex in hypergraph.vertices:
                self.assertIn(vertex, list(graph.nodes))

            edges = []
            for elem in list(graph.edges):
                edges.extend(elem)

            for edge in hypergraph.hyperedges:
                for node in edge:
                    self.assertIn(node, edges)

            self.assertEqual(len(hypergraph.hyperedges), len(hypergraph.weights))
            assert_array_equal(list(graph.nodes), hypergraph.vertices)

    def test_empty_hypergraph(self):
        # Hypergraph without hyperedges
        hyperedges = []
        vertices = ["a", "b", "c", "d"]
        vertex_labels = ["A", "B", "C", "D"]
        weights = [1, 2, 3]
        positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges, vertices, vertex_labels, weights, positions, repulse,
        )
        graph = hypergraph.complete_and_star_associated_graph()
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 0)

    def test_single_hyperedge(self):
        # one hyperedge
        hyperedges = [[1, 3]]
        vertices = [1, 2, 3, 4]
        vertex_labels = ["A", "B", "C", "D"]
        weights = [1, 1]
        positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges, vertices, vertex_labels, weights, positions, repulse,
        )
        graph = hypergraph.complete_and_star_associated_graph()
        self.assertEqual(len(graph.nodes), len(hypergraph.vertices))
        self.assertEqual(len(graph.edges), 3)

    def test_multiple_hyperedges(self):
        # multiple hyperedges
        hyperedges = [[1, 3], [2, 3, 4]]
        vertices = [1, 2, 3, 4]
        vertex_labels = ["A", "B", "C", "D"]
        weights = [1, 1]
        positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges, vertices, vertex_labels, weights, positions, repulse,
        )
        graph = hypergraph.complete_and_star_associated_graph()
        self.assertEqual(len(graph.nodes), 6)

    def test_distance_equal(self):
        # distance of edges should be equal because of equal weights
        hyperedges = [[1, 2], [2, 3], [1, 3]]
        vertices = [1, 2, 3]
        vertex_labels = [1, 2, 3]
        weights = [1, 1, 1]
        positions = np.array([[0, 0], [1, 1], [2, 2]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges, vertices, vertex_labels, weights, positions, repulse,
        )
        graph = hypergraph.complete_and_star_associated_graph()
        distance = []
        distance.append(nx.dijkstra_path_length(graph, 1, 2))
        distance.append(nx.dijkstra_path_length(graph, 2, 3))
        distance.append(nx.dijkstra_path_length(graph, 1, 3))
        self.assertEqual(distance, [distance[0]] * len(distance))

    def test_lower_distance(self):
        # distance of connected vertices should be less or equal to not connected vertices
        hyperedges = [[1, 2, 3], [2, 3]]
        vertices = [1, 2, 3, 4]
        vertex_labels = [1, 2, 3, 4]
        weights = [17]
        positions = np.array([[0, 0], [1, 1], [2, 2]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges, vertices, vertex_labels, weights, positions, repulse,
        )

        graph = hypergraph.complete_and_star_associated_graph()
        self.assertLessEqual(
            nx.dijkstra_path_length(graph, 1, 2), nx.dijkstra_path_length(graph, 1, 4)
        )

    # Two Hypergraphs with the same data should be equal
    def test_consistency(self):
        hyperedges = [[1, 2, 3], [2, 3]]
        vertices = [1, 2, 3, 4]
        vertex_labels = [1, 2, 3, 4]
        weights = [17]
        positions = np.array([[0, 0], [1, 1], [2, 2]])
        repulse = True

        hypergraph = Hypergraph(
            hyperedges,
            vertices,
            vertex_labels,
            weights,
            positions,
            repulse,
        )

        graph1 = hypergraph.complete_and_star_associated_graph()
        graph2 = hypergraph.complete_and_star_associated_graph()

        self.assertEqual(len(graph1.edges), len(graph2.edges))
        self.assertEqual(len(graph1.nodes), len(graph2.nodes))
        self.assertEqual(graph1.edges, graph2.edges)
        self.assertEqual(graph1.nodes, graph2.nodes)
        self.assertEqual(nx.dijkstra_path_length(graph1, 1, 2), nx.dijkstra_path_length(graph2, 1, 2))
        self.assertEqual(nx.dijkstra_path_length(graph1, 2,3), nx.dijkstra_path_length(graph2, 2, 3))
        self.assertTrue(isinstance(graph1, nx.Graph))
        self.assertTrue(isinstance(graph2, nx.Graph))
        for edge in graph1.edges:
            self.assertEqual(graph1.edges[edge]['weight'], graph2.edges[edge]['weight'])


if __name__ == "__main__":
    unittest.main()
