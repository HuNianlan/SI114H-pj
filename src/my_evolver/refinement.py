from Geometric_Elements import Vertex,Facet,Edge,find_vertex_by_coordinates,find_edge_by_vertices
from Geometric_Elements import VERTEXS,FACETS,EDGES
from utils import get_para
def single_facet_refinement(facet:Facet):
    """refinement by creating new vertices at the midpoints of all edges and use these to 
    subdivide each facet into four new facets each similar to the original."""
    v1 = facet.vertex1
    v2 = facet.vertex2
    v3 = facet.vertex3
    
    # Calculate midpoints of edges
    mid12 = find_vertex_by_coordinates((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)
    mid23 = find_vertex_by_coordinates((v2.x + v3.x) / 2, (v2.y + v3.y) / 2, (v2.z + v3.z) / 2)
    mid31 = find_vertex_by_coordinates((v3.x + v1.x) / 2, (v3.y + v1.y) / 2, (v3.z + v1.z) / 2)
    
    if mid12 is None:
        mid12 = Vertex((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)
        VERTEXS.append(mid12)
    if mid23 is None:
        mid23 = Vertex((v2.x + v3.x) / 2, (v2.y + v3.y) / 2, (v2.z + v3.z) / 2)
        VERTEXS.append(mid23)
    if mid31 is None:   
        mid31 = Vertex((v3.x + v1.x) / 2, (v3.y + v1.y) / 2, (v3.z + v1.z) / 2)
        VERTEXS.append(mid31)

    
    # VERTEXS.append(mid12)
    # VERTEXS.append(mid23)   
    # VERTEXS.append(mid31)
    
    EDGES.append(Edge(mid12,mid23))
    EDGES.append(Edge(mid23,mid31))
    EDGES.append(Edge(mid31,mid12))

    EDGES.append(Edge(v1, mid12)) if find_edge_by_vertices(v1, mid12) is None else None
    EDGES.append(Edge(v2, mid23)) if find_edge_by_vertices(v2, mid23) is None else None
    EDGES.append(Edge(v3, mid31)) if find_edge_by_vertices(v3, mid31) is None else None
    EDGES.append(Edge(mid12, v2)) if find_edge_by_vertices(mid12, v2) is None else None
    EDGES.append(Edge(mid23, v3)) if find_edge_by_vertices(mid23, v3) is None else None
    EDGES.append(Edge(mid31, v1)) if find_edge_by_vertices(mid31, v1) is None else None

    # Create new facets
    FACETS.append(Facet(v1, mid12, mid31))
    FACETS.append(Facet(mid12, v2, mid23))
    FACETS.append(Facet(mid31, mid23, v3))
    FACETS.append(Facet(mid12, mid23, mid31))
    
    # FACETS.remove(facet)  # Remove the original facet after refinement
    # print(f"Refined facet: {facet} into 4 new facets.")


def refinement():
    """Refine a list of facets by subdividing each facet into four new facets.
    (|V|,|E|,|F|) -> (|V|+|E|, 2|E|+3|F|, 4|F|)
    """
    n = len(FACETS)
    m = len(EDGES)
    for i in range(n):
        facet = FACETS[i]
        single_facet_refinement(facet)
    del FACETS[:n]
    del EDGES[:m]  # Clear the edges list after refinement
    get_para()  # Print the number of vertices, edges, and facets after refinement