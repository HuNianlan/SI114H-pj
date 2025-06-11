from Geometric_Elements import create_edges,create_facets,create_vertices,create_bodies,update_facet_of_body
from utils import get_para,get_body_para


def initialize(vertex_list, edge_list, face_list,body_list=None,body_cons = None):
    """Initialize the geometric elements by creating vertices, edges, and facets."""
    create_vertices(vertex_list)
    create_edges(edge_list)
    create_bodies(body_list,body_cons)  # Create bodies if provided
    create_facets(face_list)  # Create faces from the face list
    update_facet_of_body()  # Update facets of bodies after creation
    get_para()  # Print the number of vertices, edges, and facets after initialization
    for i in range(len(body_list)):
        get_body_para(i + 1)

    