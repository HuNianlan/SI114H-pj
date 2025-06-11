from math import pi, cos, sin
from init import initialize
from iterate import iterate_catenoid
from refinement import refinement
from utils import get_vertex_list1, get_facet_list1, get_facet_list,get_vertex_list
import polyscope as ps

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
body_list = []
volume_constraint = []  # 不用 volume，而是用固定边界

# -------------------- 初始化数据 --------------------
initialize(vertex_list, edge_list, face_list, body_list, volume_constraint)


iterate_catenoid(get_vertex_list1(),get_facet_list1(), num_iterations=10)
refinement()
iterate_catenoid(get_vertex_list1(),get_facet_list1(), num_iterations=15)

# 2648


########################################################################################################
# import polyscope as ps
# import numpy as np
# ps.init()
# ps.set_ground_plane_mode('none')
# ps.register_surface_mesh("Catenoid_result",np.array(get_vertex_list()),np.array(get_facet_list()))
# ps.show()

from visualization import plot_mesh

plot_mesh(get_vertex_list(), get_facet_list(), "Optimized Mesh")
