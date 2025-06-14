from math import pi, cos, sin
from iterate import iterate

# -------------------- 初始化数据 --------------------
from web import webstruct
from boundary import Boundary,Ring_XY

RMAX = 1.5088795   # minimum radius for height
ZMAX = 1.0


def bound_func1(rho):#upper ring
    x = RMAX *cos(rho[0])
    y = RMAX *sin(rho[0])
    z = ZMAX
    return [x,y,z]

def bound_func2(rho):#lower ring
    x = RMAX *cos(rho[0])
    y = RMAX *sin(rho[0])
    z = -ZMAX
    return [x,y,z]

# boundary1 = Boundary(1,bound_func1)
# boundary2 = Boundary(1,bound_func2)

boundary1 = Ring_XY([0,0],RMAX,ZMAX)
boundary2 = Ring_XY([0,0],RMAX,-ZMAX)

# -------------------- 顶点坐标（12个点，按角度计算） ----------------------
num_segments = 6

theta = [i * 2 * pi / num_segments for i in range(num_segments)] # 0 ~ pi*5/3

# 上圆
vertex_top = [[bound_func1([t]), True,boundary1] for t in theta]
# 下圆
vertex_bot = [[bound_func2([t]), True,boundary2] for t in theta]

vertex_list = vertex_top + vertex_bot  # 共12个顶点

# -------------------- 边构造（共 18 条边，来自 Evolver） --------------------
edge_list = []

# 上圆 1-6 fixed
for i in range(num_segments):
    edge_list.append([i + 1, (i + 1) % num_segments + 1,True,boundary1])

# 下圆 7-12 fixed
for i in range(num_segments):
    edge_list.append([i + 7, (i + 1) % num_segments + 7,True,boundary2])

# 连接上下圆柱
for i in range(num_segments):
    edge_list.append([i + 1, i + 7])
# -------------------- 面片构造（6 个长方形，每个转成两个三角形） --------------------
face_list = [[1,14,-7,-13],[2,15,-8,-14],[3,16,-9,-15],[4,17,-10,-16],[5,18,-11,-17],[6,13,-12,-18]]

# -------------------- 体（body） --------------------
# body_list = [[1,2,3,4,5,6]]
# volume_constraint = [1.1]  # 不用 volume，而是用固定边界
body_list = []
volume_constraint = []

from energy import Sq_Mean_Curvature
web = webstruct(vertex_list, edge_list, face_list,body_list,volume_constraint)
web.equiangulate()
web.refinement()
iterate(web, num_iterations=250)
web.delete_short_edges(0.042)
# 0.042
web.pop_vertex()
# iterate(web, num_iterations=5)
iterate(web, num_iterations=500)

########################################################################################################
from visualization import plot_mesh

plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")
