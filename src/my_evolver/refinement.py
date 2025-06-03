from Geometric_Elements import Vertex, Facet, Edge, find_vertex_by_coordinates, find_edge_by_vertices
from Geometric_Elements import VERTEXS, FACETS, EDGES
from utils import get_para

def get_or_create_midpoint(v1, v2):
    x, y, z = (v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2
    mid = find_vertex_by_coordinates(x, y, z)
    if mid is None:
        mid = Vertex(x, y, z)
        VERTEXS.append(mid)
    return mid

def get_or_create_edge(v1, v2):
    if find_edge_by_vertices(v1, v2) is None:
        EDGES.append(Edge(v1, v2))

def single_facet_refinement(facet: Facet):
    v1, v2, v3 = facet.vertex1, facet.vertex2, facet.vertex3

    # Midpoints
    mid12 = get_or_create_midpoint(v1, v2)
    mid23 = get_or_create_midpoint(v2, v3)
    mid31 = get_or_create_midpoint(v3, v1)

    # Create edges (with duplication check)
    get_or_create_edge(v1, mid12)
    get_or_create_edge(v2, mid23)
    get_or_create_edge(v3, mid31)
    get_or_create_edge(mid12, mid23)
    get_or_create_edge(mid23, mid31)
    get_or_create_edge(mid31, mid12)

    # Create new facets
    FACETS.append(Facet(v1, mid12, mid31,facet._face_id))
    FACETS.append(Facet(mid12, v2, mid23,facet._face_id))
    FACETS.append(Facet(mid31, mid23, v3,facet._face_id))
    FACETS.append(Facet(mid12, mid23, mid31,facet._face_id))

def refinement():
    n = len(FACETS)
    m = len(EDGES)

    # Copy of the original facets to avoid modifying while iterating
    original_facets = FACETS[:n]

    for facet in original_facets:
        single_facet_refinement(facet)

    del FACETS[:n]  # Remove original facets
    del EDGES[:m]   # Remove old edges

    get_para()  # Print info
