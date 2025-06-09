from Geometric_Elements import FACETS,EDGES,VERTEXS,BODIES
from Geometric_Elements import Body

import torch
edge_diff = 1
facet_diff = 1
def get_vertex_list() -> list[list[float]]:
    """Get the coordinates of all vertices."""
    return [[v.x, v.y, v.z] for v in VERTEXS]

def get_vertex_list1() ->torch.Tensor:
    """Get the coordinates of all vertices as a tensor."""
    return torch.stack([v.coord for v in VERTEXS])

def get_edge_list() -> list[list[int]]:
    """Get the list of edges as pairs of vertex IDs."""
    return [[e.vertex1.vertex_id, e.vertex2.vertex_id] for e in EDGES]

def get_facet_list() -> list[list[int]]:
    """Get the list of facets as triplets of vertex IDs."""
    return [[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in FACETS]

def get_facet_list1() ->torch.Tensor:
    """Get the list of facets as a tensor of vertex coordinates."""
    return torch.tensor([[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in FACETS], dtype=torch.int64)


def get_para():
    print(f"vertices:{len(VERTEXS)}, edges:{len(EDGES)}, facets:{len(FACETS)}")

def get_body_para(bid):
    body:Body = BODIES[bid-1]
    print(f"body {bid}: volume:{body.compute_volume()}, area:{body.get_surface_area()}")

