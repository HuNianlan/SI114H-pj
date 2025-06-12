from math import pi, cos, sin
from iterate import iterate_catenoid,iterate

# -------------------- 顶点坐标（12个点，按角度计算） ----------------------
RMAX = 1.5088795
ZMAX = 1.0
num_segments = 6

theta = [i * 2 * pi / num_segments for i in range(num_segments)] # 0 - pi*5/3

# 上圆
vertex_top = [[RMAX * cos(t), RMAX * sin(t), ZMAX, True] for t in theta]
# 下圆
vertex_bot = [[RMAX * cos(t), RMAX * sin(t), -ZMAX, True] for t in theta]

vertex_list = vertex_top + vertex_bot  # 共12个顶点

# -------------------- 边构造（共 18 条边，来自 Evolver） --------------------
edge_list = []

# 上圆 1-6 fixed
for i in range(num_segments):
    edge_list.append([i + 1, (i + 1) % num_segments + 1])

# 下圆 7-12 fixed
for i in range(num_segments):
    edge_list.append([i + 7, (i + 1) % num_segments + 7])

# 连接上下圆柱
for i in range(num_segments):
    edge_list.append([i + 1, i + 7])
# -------------------- 面片构造（6 个长方形，每个转成两个三角形） --------------------
face_list = [[1,14,-7,-13],[2,15,-8,-14],[3,16,-9,-15],[4,17,-10,-16],[5,18,-11,-17],[6,13,-12,-18]]

# -------------------- 体（body） --------------------
# body_list = [[-1,-2,-3,-4,-5,-6]]
# volume_constraint = [1.1]  # 不用 volume，而是用固定边界
body_list = []
volume_constraint = []
# -------------------- 初始化数据 --------------------
from web import webstruct
web = webstruct(vertex_list, edge_list, face_list,body_list,volume_constraint)
for i in range(1):
    iterate_catenoid(web, num_iterations=15)
    web.refinement()



########################################################################################################
# import polyscope as ps
# import numpy as np
# ps.init()
# ps.set_ground_plane_mode('none')
# ps.register_surface_mesh("Catenoid_result",np.array(get_vertex_list()),np.array(get_facet_list()))
# ps.show()

from visualization import plot_mesh

plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")
