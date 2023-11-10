"""
Module description! => For every module after restructure

Maybe link to own repo on INM-6 github

__init__py from .visualize_spade import visualize_spade
Copyright, License
"""

import holoviews as hv
from holoviews import opts
from holoviews.streams import Pipe
import numpy as np

from viziphant.patterns_src.hypergraph import Hypergraph

hv.extension('matplotlib')


class VisualizationStyle:
    INVISIBLE = 1
    NOCOLOR = 2
    COLOR = 3


class View:
    """
    Handler for the visualization of hypergraphs.
    This class constructs a visualization of the hypergraphs
    and interactive widgets to alter the visualization.
    These are combined into a single layout which is then displayed
    using the `show` method.
    In summary, this class represents an interactive tool
    for the visualization of hypergraphs.
    """

    def __init__(self, hypergraphs, title=None):
        """
        Constructs a View object that handles the visualization
        of the given hypergraphs.

        Parameters
        ----------
        hypergraphs: list of Hypergraph objects
            Hypergraphs to be visualized.
            Each hypergraph should contain data of one data set.
        """

        # Hyperedge drawings
        self.polygons = []

        # Which color of the color map to use next
        self.current_color = 1

        # Size of the vertices
        self.node_radius = 0.2

        # Selected title of the figure
        self.title = title

        # If no data was provided, fill in dummy data
        if hypergraphs:
            self.hypergraphs = hypergraphs
        else:
            self.hypergraphs = [Hypergraph(hyperedges=[], vertices=[])]

        # Extract all vertices and corresponding labels
        # from the list of hypergraphs
        # Mapping is preserved
        self.vertices, self.vertex_labels = Hypergraph.get_union_of_vertices(
            self.hypergraphs)

        # In the beginning, no positions are available
        # These are computed when necessary,
        # either when explicitly requested by the user
        # or when implicitly required to display the visualization
        self.positions = None

        self.n_hypergraphs = len(self.hypergraphs)

        # Set up the visualizations and interaction widgets that need to be
        # displayed
        self.dynamic_map, self.pipe = self._setup_graph_visualization()
        self.dynamic_map_edges, self.edges_pipe = \
            self._setup_hyperedge_drawing()

        self.plot = None

    def _setup_graph_visualization(self):
        """
        Set up the holoviews DynamicMap object
        that visualizes the nodes of a graph
        Returns
        -------
        dynamic_map: hv.DynamicMap
            The DynamicMap object in which the nodes will be visualized.
            This object needs to be displayed.
        pipe: Pipe
            The pipe which new data (i.e., new and changed nodes) are sent into
        """

        # Pipe that gets the updated data that are then sent to the dynamic map
        pipe = Pipe(data=[], memoize=True)

        # Holoviews DynamicMap with a stream that gets data from the pipe
        # The hv.Graph visualization is used for displaying the data
        # hv.Graph displays the nodes (and optionally binary edges) of a graph
        dynamic_map = hv.DynamicMap(hv.Graph, streams=[pipe])

        # Define options for visualization
        dynamic_map.opts(
            # Some space around the Graph in order to avoid nodes being on the
            # edges of the visualization
            padding=0.5,
            # # Interactive tools that are provided by holoviews
            # tools=['box_select', 'lasso_select', 'tap', 'hover'],
            # Do not show axis information (i.e., axis ticks etc.)
            xaxis=None, yaxis=None
        ).opts(opts.Graph(
            # Where to get information on color
            # TODO
            # color_index='index',
            # All in black
            cmap=['#ffffff', '#ffffff'] * 50,
            # Size of the nodes
            node_size=self.node_radius))

        return dynamic_map, pipe

    def _setup_hyperedge_drawing(self):
        """
        Set up the holoviews DynamicMap object
        that visualizes the hyperedges of a hypergraph
        Returns
        -------
        dynamic_map: hv.DynamicMap
            The DynamicMap object in which the hyperedges will be visualized.
            This object needs to be displayed.
        pipe: Pipe
            The pipe which new data (i.e., new and changed hyperedges)
            are sent into
        """

        # Pipe that gets the updated data that are then sent to the dynamic map
        pipe = Pipe(data=[], memoize=True)

        # Function that creates hv.Polygons from the defined points
        # Every hyperedge drawing is one (or multiple for triangulation) of
        # these polygons
        def create_polygon(*args, **kwargs):
            # Define holoviews polygons with additional metadata dimensions:
            # value is the index into the color map
            # alpha specifies the alpha value for the fill color
            # line_alpha specifies the alpha value for the boundary
            pol = hv.Polygons(*args,
                              vdims=['value', 'alpha'],
                              **kwargs)
            # Define the mapping described above
            pol.opts(alpha='alpha')
                     # TODO: check this
                     #line_alpha='line_alpha',
                     # color_index='value')
            # The polygons are then displayed in the DynamicMap object
            return pol

        # dynamic_map gets input from pipe and visualizes it as polygons using
        # the create_polygon function
        dynamic_map = hv.DynamicMap(create_polygon, streams=[pipe])

        if self.n_hypergraphs <= 1:
            # If there is only a single hypergraph, all hyperedges are colored
            # differently
            import colorcet
            cmap = colorcet.glasbey[:len(self.hypergraphs[0].hyperedges)]
        elif self.n_hypergraphs <= 10:
            # Select Category10 colormap as default for up to 10 data sets
            # This is an often used colormap
            from bokeh.palettes import all_palettes
            cmap = list(all_palettes['Category10'][10][1:self.n_hypergraphs+1])[::-1]
        else:
            # For larger numbers of data sets, select Glasbey colormap
            import colorcet
            cmap = colorcet.glasbey[:self.n_hypergraphs]

        # Setting limits for colormaps to make sure color index
        # (hypergraph or hyperedge index) is used like a list index
        # Generally indexing is equally spaced depending on the indexes
        # that actually occur, e.g., if cmap has 5 entries and only indices
        # 1, 2, 3 occur, colors 1, 3 and 5 will be used due to equal spacing
        # Desired behavior here is always using colors 1, 2 and 3
        # for indices 1, 2 and 3.
        # Setting the limit to 5 in the above example causes the colormap
        # to be spaced from 1 to 5, mapping color 1 to index 1 and
        # color 5 to index 5.
        # Equal spacing in between makes sure all other indices
        # are mapped correctly as well.

        # If there is more than one hypergraph, one color per hypergraph is
        # needed
        if self.n_hypergraphs > 1:
            dynamic_map.opts(cmap=cmap, clim=(1, self.n_hypergraphs))
        # If there is only one hypergraph, one color per hyperedge is needed
        else:
            dynamic_map.opts(cmap=cmap,
                             clim=(1, len(self.hypergraphs[0].hyperedges)))

        return dynamic_map, pipe

    def show(self,
             subset_style=VisualizationStyle.COLOR,
             triangulation_style=VisualizationStyle.INVISIBLE):
        """
        Set up the correct arrangement and combination of the
        DynamicMap objects.
        """

        # Positions of nodes are required for drawing
        # Thus call layout algorithm if no positions are available
        if not isinstance(self.positions, np.ndarray) and not self.positions:
            self.layout_hypergraph()
        # Pass positions of nodes to the visualization
        self._update_nodes([[[], []], [self.positions, self.vertices]])
        # Call the handler that manages visualization of hyperedges
        self.draw_hyperedges(subset_style=subset_style,
                             triangulation_style=triangulation_style)

        # Visualization as an overlay of the graph visualization and the
        # hyperedge drawings
        plot = self.dynamic_map * self.dynamic_map_edges
        # Set size of the plot to a square to avoid distortions
        self.plot = plot.redim.range(x=(-1, 11), y=(-1, 11))

        return hv.render(plot, backend="matplotlib")

    def draw_hyperedges(self,
                        subset_style=VisualizationStyle.COLOR,
                        triangulation_style=VisualizationStyle.INVISIBLE):
        """
        Handler for drawing the hyperedges of the hypergraphs
        in the given style

        Parameters
        ----------
        subset_style: enum VisualizationStyle
            How to do the subset standard visualization of the hyperedges:
            VisualizationStyle.INVISIBLE => not at all
            VisualizationStyle.NOCOLOR => Only contours without color
            VisualizationStyle.COLOR => Contours filled with color
        triangulation_style: enum VisualizationStyle
            How to do the triangulation visualization of the hyperedges:
            VisualizationStyle.INVISIBLE => not at all
            VisualizationStyle.NOCOLOR => Only contours without color
            VisualizationStyle.COLOR => Contours filled with color
        """

        # Collect all polygons to pass them to the visualization together
        polygons = []
        # Color is incremented per hyperedge if there is only 1 hypergraph
        self.current_color = 1
        # Create polygon for every hyperedge in every hypergraph
        for i, hypergraph in enumerate(self.hypergraphs, start=1):
            for hyperedge in hypergraph.hyperedges_indexes:
                polygons_subset = []
                polygons_triang = []
                # Whether and how to show subset standard and
                # triangulation method
                # Options are: invisible, nocolor, color
                # If invisible, the corresponding visualization does not need
                # to be constructed
                if subset_style != VisualizationStyle.INVISIBLE:
                    polygons_subset.append(create_subset(hyperedge,
                                                         self.positions,
                                                         self.node_radius))
                if triangulation_style != VisualizationStyle.INVISIBLE:
                    polygons_triang.extend(
                        create_triangulation(hyperedge, self.positions))

                # Postprocessing: Adding color ('value')
                # and additional information
                # to the polygons representing the hyperedges
                def process_polygons(polygons, color=True):
                    for polygon in polygons:
                        if not color:
                            polygon['value'] = -1
                        elif self.n_hypergraphs > 1:
                            polygon['value'] = i
                        else:
                            polygon['value'] = self.current_color
                            self.current_color += 1
                        # Required for selection widgets, sorting by these
                        # properties happens later on
                        polygon['length'] = len(hyperedge)
                        polygon['type'] = i
                        polygon['vertices'] = self.vertices[hyperedge]
                    return polygons

                # Call the postprocessing function, color added only if needed
                polygons.extend(process_polygons(
                        polygons_subset,
                        subset_style == VisualizationStyle.COLOR))
                polygons.extend(process_polygons(
                        polygons_triang,
                        triangulation_style == VisualizationStyle.COLOR))

        # Save as instance attribute so widgets can work with the polygons
        self.polygons = polygons
        # Send all polygons to the visualization
        # There they will be preselected depending on their properties and then
        # visualized

        self._update_hyperedge_drawings(self.polygons)

    def layout_hypergraph(self):
        """
        Helper function that gets a layout for a hypergraph containing
        all given information from all hypergraphs and visualizes it.
        A hypergraph is constructed that contains all vertices and
        all hyperedges of all given hypergraphs.
        A layout for this hypergraph is computed as defined in
        class Hypergraph.
        This layout is saved as an instance variable and passed to the
        visualization of the nodes.
        """

        # Construct a single hypergraph from all hypergraphs for layouting
        # Union of all hyperedges
        hyperedges = [x for hg in self.hypergraphs for x in hg.hyperedges]
        # Weights treated analogously
        weights = [x for hg in self.hypergraphs for x in hg.weights]
        hypergraph = Hypergraph(vertices=list(self.vertices),
                                hyperedges=hyperedges,
                                weights=weights,
                                repulse=repulsive)
        # Get layout
        self.positions = hypergraph.layout_hypergraph()
        # Visualize
        self._update_nodes(data=[[[], []], [self.positions, self.vertices]])

    # Update the view
    def _update_hyperedge_drawings(self, data):
        """

        Parameters
        ----------
        data: list of dicts
            Every dict represents one polygon that will be visualized.
            Required keys are:
                'x': x coordinates of the polygon's vertices
                'y': y coordinates of the polygon's vertices
                'alpha': alpha value for fill color
                'line_alpha': alpha value for line/stroke color
                'length': size of the corresponding hyperedge
                'type': index of the corresponding hypergraph
                'vertices': indices of the hypergraph vertices
                    of the corresponding hyperedge
                'value': index into color map indicating which color to use
                    for this polygon
        """

        for x in data:
            x['alpha'] = 0.2
            if x['value'] == -1:
                x['alpha'] = 0.0
                x['line_alpha'] = 1.0
            else:
                x['line_alpha'] = 0.2

        self.edges_pipe.send(data=data)

    def _update_nodes(self, data):
        """
        Updates the visualization of the nodes based on new x and y coordinates
        and metadata.

        Parameters
        ----------
        data: list
            The data containing the new positions of the vertices
        """

        vertex_ids = data[1][1]

        # TODO see if library function in holoviews is available with option to
        # display edges or not
        try:
            self.positions = data[1][0]
            pos_x = data[1][0].T[0]
            pos_y = data[1][0].T[1]
            edge_source = data[0][0]
            edge_target = data[0][1]
        except IndexError:
            pos_x = []
            pos_y = []
            edge_source = []
            edge_target = []
            vertex_ids = []

        vertex_labels = self.vertex_labels
        nodes = hv.Nodes((pos_x, pos_y, vertex_ids, vertex_labels),
                         extents=(0.01, 0.01, 0.01, 0.01),
                         vdims='Label')

        new_data = ((edge_source, edge_target), nodes)
        self.pipe.send(new_data)


# Parameters heuristically tested to produce pleasing results
# These need not be changeable for the user but might be optimized

# Weight set for actual edges when making a complete graph
# out of a non-weighted graph.
# Weights for edges that do not exist in the original graph are 1.
weight = 50
# Whether vertices should repel each other more strongly
# if they are in overlapping hyperedges
# but not in the same hyperedge
repulsive = False


def create_triangulation(hyperedge, positions):
    """
    Construction of triangles resulting from Delaunay triangulation of the
    vertices of the hyperedge.
    The union of all triangles represents the Delaunay triangulation
    and is used for visualization of a hyperedge.

    Parameters
    ----------
    hyperedge: list or np.ndarray
        Vertices of the hyperedge given as indices for the positions array.
        Vertices need to be given such that for
        every vertex v in hyperedge: positions[v] is the position of vertex v.
    positions: np.ndarray
        Positions of all vertices as given by a layout algorithm

    Returns
    -------
    triangles: list of dict
        Multiple triangles that constitute the Delaunay triangulation of the
        vertices of the hyperedge
        'x': x coordinates of the triangle's vertices
        'y': y coordinates of the triangle's vertices
    """

    triangles = []
    # Delaunay triangulation algorithm is only required for more than 3 points
    # (i.e., vertices)
    if len(hyperedge) > 3:
        # Delaunay triangulation algorithm as implemented in scipy.spatial,
        # uses the QHull C library underneath
        from scipy.spatial import Delaunay
        # TODO: Force all to be a vertex of triangles, exclude coplanarity
        # (QHull options)
        triangulation = Delaunay(positions[hyperedge])
        # Every simplex (i.e., triangle) of the triangulation becomes one
        # polygon for holoviews to draw
        for triangle in positions[hyperedge][triangulation.simplices]:
            triangles.append(dict(x=triangle[:, 0], y=triangle[:, 1]))
    # For at most 3 points, simply take all points as vertices of a polygon
    else:
        triangles.append(
            dict(x=positions[hyperedge][:, 0], y=positions[hyperedge][:, 1]))

    return triangles


def create_subset(hyperedge, positions, node_radius):
    """
    Construction of a smooth shape surrounding all vertices of the
    given hyperedge using Catmull-Rom spline curve
    interpolation.

    Parameters
    ----------
    hyperedge: list or np.ndarray
        Vertices of the hyperedge given as indices for the positions array.
        Vertices need to be given such that for
        every vertex v in hyperedge: positions[v] is the position of vertex v.
    positions: np.ndarray
        Positions of all vertices as given by a layout algorithm
    node_radius: float
        Radius of the circle of the node depicting a vertex

    Returns
    -------
    polygon: dict
        Polygon representing the subset shape sampled from the interpolating
        spline curve.
        'x': np.ndarray of x coordinates of the points describing the polygon
        'y': np.ndarray of y coordinates of the points describing the polygon
    """

    # Construct control points using corresponding function
    # Special constructions are necessary for hyperedges with 1 or 2 neurons
    if len(hyperedge) == 1:
        control_points = _construct_subset_unary(
            hyperedge, positions, node_radius)
    elif len(hyperedge) == 2:
        control_points = _construct_subset_binary(
            hyperedge, positions, node_radius)
    else:
        control_points = _construct_subset_larger(
            hyperedge, positions, node_radius)

    # Construction of a polygon using spline interpolation
    polygon = {'x': [], 'y': []}

    # If there are n control points, there are n segments
    # between control points due to circular arrangement
    # For every segment the interpolating spline curve is computed
    for i in range(len(control_points)):
        # 4 vertices are necessary for interpolation of current segment
        vertices = list(range(-3 + i, i + 1))
        vertices = [x % len(control_points) for x in vertices]

        # Catmull-Rom spline curve interpolation of the control points
        # 4 control points are necessary for every segment
        seg = _catmull_rom(control_points[vertices].T)

        # Add new segment to the polygon
        polygon['x'] = np.concatenate((polygon['x'], seg[0]))
        polygon['y'] = np.concatenate((polygon['y'], seg[1]))

    return polygon


def _construct_subset_unary(hyperedge, positions, node_radius):
    """
    Construction of control points for subset standard for unary hyperedges

    Parameters
    ----------
    hyperedge: list or np.ndarray
        Vertices of the hyperedge given as indices for the positions array.
        Vertices need to be given such that for
        every vertex v in hyperedge: positions[v] is the position of vertex v.
    positions: np.ndarray
        Positions of all vertices as given by a layout algorithm
    node_radius: float
        Radius of the circle of the node depicting a vertex

    Returns
    -------
    control_points: np.ndarray
        2D array containing positions of control points
        in counterclockwise order
    """
    pos = positions[hyperedge]
    # Circular arrangement of control points around node
    #    x
    # x  o  x
    #    x
    # Arranged in counterclockwise order
    control_points = np.ndarray((4, 2))
    control_points[0] = pos + node_radius * 3 * np.array([0, 1])
    control_points[1] = pos + node_radius * 3 * np.array([-1, 0])
    control_points[2] = pos + node_radius * 3 * np.array([0, -1])
    control_points[3] = pos + node_radius * 3 * np.array([1, 0])

    return control_points


def _construct_subset_binary(hyperedge, positions, node_radius):
    """
    Construction of control points for subset standard for binary hyperedges

    Parameters
    ----------
    hyperedge: list or np.ndarray
        Vertices of the hyperedge given as indices for the positions array.
        Vertices need to be given such that for
        every vertex v in hyperedge: positions[v] is the position of vertex v.
    positions: np.ndarray
        Positions of all vertices as given by a layout algorithm
    node_radius: float
        Radius of the circle of the node depicting a vertex

    Returns
    -------
    control_points: np.ndarray
        2D array containing positions of control points
        in counterclockwise order
    """
    pos = positions[hyperedge]
    node1 = pos[0]
    node2 = pos[1]

    diff = pos[1] - pos[0]
    # Difference vector scaled to the desired distance between node center
    # and control point for interpolation
    # Control point will be constructed by moving diff away from actual node
    # center
    diff = diff / np.linalg.norm(diff) * node_radius * 2

    # Normal vector (orthogonal to difference vector), scaled analogously
    orth = diff[::-1].copy()
    orth[1] = -orth[1]

    # Construction of control points:
    #   x             x
    # x o             o x
    #   x             x
    # o represents a node, x a control point
    # Ordered counterclockwise for interpolation
    control_points = np.ndarray((6, 2))
    # Control point left of left node (example based on drawing above)
    control_points[0] = pos[0] - diff
    # Control points below nodes
    control_points[1] = node1 + orth
    control_points[2] = node2 + orth
    # Control point right of right node
    control_points[3] = pos[1] + diff
    # Control points above nodes
    control_points[4] = node2 - orth
    control_points[5] = node1 - orth

    return control_points


def _construct_subset_larger(hyperedge, positions, node_radius):
    """
    Construction of control points for subset standard for hyperedges
    containing more than 2 vertices

    Parameters
    ----------
    hyperedge: list or np.ndarray
        Vertices of the hyperedge given as indices for the positions array.
        Vertices need to be given such that for
        every vertex v in hyperedge: positions[v] is the position of vertex v.
    positions: np.ndarray
        Positions of all vertices as given by a layout algorithm
    node_radius: float
        Radius of the circle of the node depicting a vertex

    Returns
    -------
    control_points: np.ndarray
        2D array containing positions of control points
        in counterclockwise order
    """

    # Order positions counterclockwise
    # Required to compute outtangent intersections for neighboring vertices
    from scipy.spatial import ConvexHull
    hull = ConvexHull(positions[hyperedge])
    pos = positions[hyperedge][hull.vertices]

    # Compute control points as intersections of outtangents
    control_points = _outtangent_intersections(
        hyperedge, positions, node_radius, hull)

    # Difference vector of node position to control point position
    diff = control_points - pos
    # Euclidean distance of node to control point
    diff_len = np.linalg.norm(diff, axis=1)
    # Normalized difference vectors
    diff = diff / diff_len[:, np.newaxis]

    # Threshold for minimal distance
    # For nodes with distances less than 1.3*node_radius, move control point
    # away to make distance 1.3*node_radius
    mask_close = diff_len < 1.3 * node_radius
    control_points[mask_close] = pos[mask_close] + \
        1.3 * node_radius * diff[mask_close]

    # Threshold for maximal distance
    # For nodes with distance larger than 3*node_radius, move control point
    # closer to make distance 3*node_radius
    mask = diff_len > 3 * node_radius
    control_points[mask] = pos[mask] + 3 * node_radius * diff[mask]

    # In case of exceeded maximal distance, an auxiliary construction
    # with additional control points is needed
    # Else the resulting subset curve would intersect the node due to a very
    # acute angle that caused threshold crossing
    num_lim = np.count_nonzero(mask)
    # 2 additional control points per threshold crossing
    new_point_pos = np.ndarray((control_points.shape[0] + 2 * num_lim, 2))
    # index for new position, may be incremented multiple times per iteration
    j = 0
    for i in range(len(control_points)):
        # If the distance does not cross the threshold for this control point
        # (this node, respectively),
        # Just use that control point
        if not mask[i]:
            new_point_pos[j] = control_points[i]
            j += 1
        # If threshold is crossed, add 2 additional control points around the
        # ode
        else:
            # Neighboring vertices, circular arrangement
            vertices = list(range(i - 1, i + 2))
            vertices = [x % len(control_points) for x in vertices]
            # Construct directions of tangents (segments of the boundary of the
            # convex hull) adjacent to i-th vertex
            dir1 = control_points[vertices][0] - control_points[vertices][1]
            dir2 = control_points[vertices][2] - control_points[vertices][1]
            # pX vectors are orthogonal to dirX, facing outward (with respect
            # to the convex hull)
            p2 = _projection(dir1, dir2) * node_radius
            p1 = _projection(dir2, dir1) * node_radius
            # One control point is 2*node_radius away from node center, moving
            # away orthogonally to dir1
            new_point_pos[j] = pos[i] + 2 * p1
            # Outtangent intersection that was already moved closer
            new_point_pos[j + 1] = control_points[i]
            # One control point is 2*node_radius away from node center, moving
            # away orthogonally to dir2
            new_point_pos[j + 2] = pos[i] + 2 * p2
            # 3 points added to array
            j += 3

    return new_point_pos


def _projection(dir_from, dir_to):
    """
    Construct a vector orthogonal to dir_to that points outward
    with respect to the angle between dir_from and dir_to.
    Placing the vector such that their starting point is the origin,
    taking the angle between them that is less than
    180ª as the inside, shifting the starting point of dir_to by the
    resulting vector will move it outward, i.e., away from dir_from,
    not causing an intersection.

    Parameters
    ----------
    dir_from: np.ndarray
        Vector that will be projected onto dir_to
    dir_to: np.ndarray
        Vector that dir_from is projected onto
    Returns
    -------
    diff: np.ndarray
        Unit vector orthogonal to dir_to, pointing from dir_from
        to the projection of dir_from onto dir_to
    """

    # Orthogonal projection from dir_from onto dir_to
    projection = dir_to / (np.linalg.norm(dir_to) ** 2) * \
        np.dot(dir_from, dir_to)
    # Subtract dir_from to construct a vector diff such that dir_from + diff =
    # projection
    diff = projection - dir_from
    # Normalize
    diff = diff / np.linalg.norm(diff)

    return diff


def _outtangent_intersections(hyperedge, positions, node_radius, hull):
    """
    Construction of outtangents by moving line segments outward
    using orthogonal projections
    Their intersections become control points for spline interpolation
    See Arafat and Bressan 2017, Hypergraph Drawing by Force-Directed Placement
    https://doi.org/10.1007/978-3-319-64471-4_31

    Parameters
    ----------
    hyperedge: np.ndarray or list
        The hyperedge based on a 0-based range representation of vertices
        in order to use it for indexing into the positions array
    positions: np.ndarray
        Positions of all vertices
    node_radius: float
        Size of a node drawing, used to determine how far to move outtangents
        outward
    hull: scipy.spatial.qhull.ConvexHull
        Convex hull of the hyperedge

    Returns
    -------
    intersections: np.ndarray
        Array of outtangent intersection points
        (in counterclockwise order in the 2D case)
    """

    # Array containing the final intersections
    intersections = np.ndarray((len(hull.vertices), 2))
    for i in range(len(hull.vertices)):
        # Select 3 neighboring indices for vertices,
        # in convex hull vertices are ordered such that neighboring indices
        # are actually neighboring vertices along the boundary
        # of the convex hull
        # Assuming circular arrangement of the vertices as given
        # by the convex hull algorithm
        # => in [0, 1, 2, 3, 4], 0 and 4 are neighbors etc.
        # Starting by constructing outtangent intersection close to vertex with
        # index 0
        current_vertex_indices = list(range(i - 1, i + 2))
        current_vertex_indices = [x % len(hull.vertices)
                                  for x in current_vertex_indices]

        # Get the actual vertices belonging the indices from above
        # Necessary, because in the convex hull not all vertices must be
        # included and order is changed to arrange vertices counterclockwise
        current_vertices = hull.vertices[current_vertex_indices]
        # Positions belonging to the actual current vertices
        pos = positions[hyperedge][current_vertices]

        # Centers of the line segments defined by the center vertices and one
        # of the outer vertices each
        c1 = (pos[0] + pos[1]) / 2
        c2 = (pos[1] + pos[2]) / 2
        # Directions of the lines defined by the center vertices and
        # one of the outer vertices each
        # e.g., x(mu) = c1 + mu*dir1
        dir1 = pos[0] - pos[1]
        dir2 = pos[2] - pos[1]

        # pX is a vector orthogonal to dirX that moves the line defined by cX
        # and dirX outward
        p2 = _projection(dir1, dir2) * node_radius
        p1 = _projection(dir2, dir1) * node_radius

        # Move lines outward by moving their centers
        c1 += p1
        c2 += p2

        # Linear system of equations to find the intersection of the lines
        # c1 + mu * dir1 = c2 + lambda * dir2 <==> (-mu) * dir1 + lambda * dir2
        # = c1 - c2
        A = np.ndarray((2, 2))
        A[:, 0] = dir2
        A[:, 1] = dir1
        b = c1 - c2
        mu = -np.linalg.solve(A, b)[1]

        # Point of intersection
        intersect = c1 + mu * dir1

        # Add to the other intersections
        intersections[i] = intersect

    return intersections


def _catmull_rom(points, n_eval=100):
    """
    Cubic Catmull-Rom spline curve interpolation using
    centripetal parameterization.
    A spline segment between the second and third given points is constructed
    and evaluated at equally spaced point in parameter space.

    See Yuksel et al. 2011, Parameterization and applications of
    Catmull-Rom curves
    https://doi.org/10.1016/j.cad.2010.08.008

    Parameters
    ----------
    points: np.ndarray
        Array of shape (4, 2) containing positions of 4 support points.
        An interpolating spline segment will be computed
        between points[1] and points[2]. points[0] and points[3] are used
        as additional control points to construct a differentiable curve.
    n_eval:int
        Number of points at which the spline segment will be evaluated.
        These points will be equally spaced between points[1] and points[2]
        in parameter space.
    Returns
    -------
    C: np.ndarray
        Array of shape (n_eval, 2), spline evaluated at n_eval equally spaced
        points in parameter space.
    """

    # t contains the parametric values for which f(t_i) = (x_i, y_i)
    # Used here is the centripetal parameterization (see Yuksel et al., 2011)
    t = np.ndarray((4,))
    t[0] = 0
    for i in range(1, 4):
        diff = points[:, i] - points[:, i - 1]
        t[i] = np.power(np.linalg.norm(diff), 0.5) + t[i - 1]

    # Points in parameter space at which the spline will be evaluated
    # 4 control points with parametric values t_0 to t_3
    # are used to determine the spline
    # Between the second and the third point (t_1 and t_2) the curve is
    # evaluated
    eval_points = np.linspace(t[1], t[2], n_eval)

    # Implementation following Yuksel et al., 2011, simplified schema
    A3 = (t[3] - eval_points[np.newaxis, :]) / (t[3] - t[2]) * points[:, 2, np.newaxis] + \
         (eval_points[np.newaxis, :] - t[2]) / (t[3] - t[2]) * points[:, 3, np.newaxis]
    A2 = (t[2] - eval_points[np.newaxis, :]) / (t[2] - t[1]) * points[:, 1, np.newaxis] + \
         (eval_points[np.newaxis, :] - t[1]) / (t[2] - t[1]) * points[:, 2, np.newaxis]
    A1 = (t[1] - eval_points[np.newaxis, :]) / (t[1] - t[0]) * points[:, 0, np.newaxis] + \
         (eval_points[np.newaxis, :] - t[0]) / (t[1] - t[0]) * points[:, 1, np.newaxis]

    B2 = (t[3] - eval_points[np.newaxis, :]) / (t[3] - t[1]) * A2 + \
         (eval_points[np.newaxis, :] - t[1]) / (t[3] - t[1]) * A3
    B1 = (t[2] - eval_points[np.newaxis, :]) / (t[2] - t[0]) * A1 + \
         (eval_points[np.newaxis, :] - t[0]) / (t[2] - t[0]) * A2

    # C contains the resulting points of evaluating the spline curve at all
    # points in eval_points
    C = (t[2] - eval_points[np.newaxis, :]) / (t[2] - t[1]) * B1 + \
        (eval_points[np.newaxis, :] - t[1]) / (t[2] - t[1]) * B2

    return C

