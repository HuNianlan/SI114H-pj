from Geometric_elements import Vertex,Facet
from Geometric_elements import VERTEXS, EDGES, FACETS

def single_facet_refinement(facet:Facet):
    """refinement by creating new vertices at the midpoints of all edges and use these to 
    subdivide each facet into four new facets each similar to the original."""
    v1 = facet.vertex1
    v2 = facet.vertex2
    v3 = facet.vertex3
    
    # Calculate midpoints of edges
    mid12 = Vertex((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)
    mid23 = Vertex((v2.x + v3.x) / 2, (v2.y + v3.y) / 2, (v2.z + v3.z) / 2)
    mid31 = Vertex((v3.x + v1.x) / 2, (v3.y + v1.y) / 2, (v3.z + v1.z) / 2)
    
    VERTEXS.append(mid12)
    VERTEXS.append(mid23)   
    VERTEXS.append(mid31)
    # Create new facets
    FACETS.append(Facet(v1, mid12, mid31))
    FACETS.append(Facet(mid12, v2, mid23))
    FACETS.append(Facet(mid31, mid23, v3))
    FACETS.append(Facet(mid12, mid23, mid31))
    
    # FACETS.remove(facet)  # Remove the original facet after refinement
    # print(f"Refined facet: {facet} into 4 new facets.")


def refinement():
    """Refine a list of facets by subdividing each facet into four new facets."""
    n = len(FACETS)
    for i in range(n):
        facet = FACETS[i]
        single_facet_refinement(facet)
    del FACETS[:n]
    # for facet in FACETS:
    #     single_facet_refinement(facet)