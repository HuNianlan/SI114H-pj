import global_state
from Geometric_Elements import Body
import numpy as np
import torch
from Geometric_Elements import Vertex,Facet


def get_vertex_list() -> list[list[float]]:
    """Get the coordinates of all vertices."""
    return [[v.x, v.y, v.z] for v in global_state.VERTEXS]

def get_vertex_list1() ->torch.Tensor:
    """Get the coordinates of all vertices as a tensor."""
    if len(global_state.VERTEXS)==0:return None
    return torch.stack([v.coord for v in global_state.VERTEXS])


def get_vertex_mask()->torch.Tensor:
    return  [[1-v.is_fixed, 1-v.is_fixed, 1-v.is_fixed] for v in global_state.VERTEXS]


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



def compute_normal(v1:Vertex, v2:Vertex, v3:Vertex):
    # 计算两个边向量
    vec1 = v2.coord - v1.coord
    vec2 = v3.coord - v1.coord
    # 叉积得到法线向量
    normal = torch.cross(vec1, vec2)
    # 单位化（归一化）
    norm = torch.linalg.norm(normal)
    if norm == 0:
        return None  # 退化三角形
    return normal / norm

def angle_between_facets(facet1:Facet, facet2:Facet):
    # 提取顶点
    n1 = compute_normal(facet1.vertex1, facet1.vertex2, facet1.vertex3)
    n2 = compute_normal(facet2.vertex1, facet2.vertex2, facet2.vertex3)

    if n1 is None or n2 is None:
        return None  # 法线无效（退化三角形）

    # 点积计算夹角的余弦值
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)  # clip防止精度误差超界
    angle_rad = np.arccos(cos_theta)  # 弧度值
    angle_deg = np.degrees(angle_rad)  # 角度值（可选）
    
    return angle_deg  # 或 angle_deg

