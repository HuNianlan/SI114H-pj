from Geometric_Elements import create_edges,create_facets,create_vertices,create_bodies
from utils import get_para,get_body_para


def initialize(vertex_list, edge_list, face_list,body_list=None):
    """Initialize the geometric elements by creating vertices, edges, and facets."""
    create_vertices(vertex_list)
    create_edges(edge_list)
    create_bodies(body_list)  # Create bodies if provided
    create_facets(face_list)  # Create faces from the face list

    get_para()  # Print the number of vertices, edges, and facets after initialization
    get_body_para(1)  # Print the parameters of the first body if it exists