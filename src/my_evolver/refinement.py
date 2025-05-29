from Geometric_elements import Vertex,Facet
from Geometric_elements import VERTEXS, EDGES, FACETS

def single_facet_refinement(facet:Facet) -> list[Facet]:
    """refinement by creating new vertices at the midpoints of all edges and use these to 
    subdivide each facet into four new facets each similar to the original."""
    v1 = facet.vertex1
    v2 = facet.vertex2
    v3 = facet.vertex3
    
    # Calculate midpoints of edges
    mid12 = Vertex((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)
    mid23 = Vertex((v2.x + v3.x) / 2, (v2.y + v3.y) / 2, (v2.z + v3.z) / 2)
    mid31 = Vertex((v3.x + v1.x) / 2, (v3.y + v1.y) / 2, (v3.z + v1.z) / 2)
    
    # Create new facets
    return [
        Facet(v1, mid12, mid31),
        Facet(mid12, v2, mid23),
        Facet(mid31, mid23, v3),
        Facet(mid12, mid23, mid31)
    ]

def refinement(facets:list[Facet]) -> list[Facet]:
    """Refine a list of facets by subdividing each facet into four new facets."""
    refined_facets = []
    for facet in facets:
        refined_facets.extend(single_facet_refinement(facet))
    return refined_facets