from Geometric_Elements import FACETS,EDGES,VERTEXS,BODIES
from Geometric_Elements import Body

def get_para():
    print(f"vertices:{len(VERTEXS)}, edges:{len(EDGES)}, facets:{len(FACETS)}")

def get_body_para(bid):
    body:Body = BODIES[bid-1]
    print(f"body {bid}: volume:{body.compute_volume()}, area:{body.get_surface_area()}")