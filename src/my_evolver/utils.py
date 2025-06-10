import global_state
from Geometric_Elements import Body
import numpy as np
import torch

def get_vertex_list() -> list[list[float]]:
    """Get the coordinates of all vertices."""
    return [[v.x, v.y, v.z] for v in global_state.VERTEXS]

def get_vertex_list1() ->torch.Tensor:
    """Get the coordinates of all vertices as a tensor."""
    if len(global_state.VERTEXS)==0:return None
    return torch.stack([v.coord for v in global_state.VERTEXS])

def get_edge_list() -> list[list[int]]:
    """Get the list of edges as pairs of vertex IDs."""
    return [[e.vertex1.vertex_id, e.vertex2.vertex_id] for e in global_state.EDGES]

def get_facet_list() -> list[list[int]]:
    """Get the list of facets as triplets of vertex IDs."""
    return [[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in global_state.FACETS]

def get_facet_list1() ->torch.Tensor:
    """Get the list of facets as a tensor of vertex coordinates."""
    return torch.tensor([[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in global_state.FACETS], dtype=torch.int64)

def get_body_list1()->torch.Tensor:
    """Get the list of bodies as a tensor of facet coordinates"""
    if len(global_state.BODIES)==0:return None
    return torch.stack([torch.tensor(np.array(b.get_facet_list())-global_state.facet_diff) for b in global_state.BODIES])

def get_para():
    print(f"vertices:{len(global_state.VERTEXS)}, edges:{len(global_state.EDGES)}, facets:{len(global_state.FACETS)}")

def get_body_para(bid):
    body:Body = global_state.BODIES[bid-1]
    print(f"body {bid}: volume:{body.compute_volume()}, area:{body.get_surface_area()}")

