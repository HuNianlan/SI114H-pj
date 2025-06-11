from utils import get_facet_list1,get_vertex_list1,get_facet_list,get_vertex_list
from refinement import refinement
from init import initialize
from energy import Area,Energy
from constraint import Volume,Constraint
from Geometric_Elements import update_vertex_coordinates
from iterate import iterate
import global_state
########################################################################################################
vertex_list = [[0.0,0.0,0.0],
               [1.0,0.0,0.0],
               [1.0,1.0,0.0],
               [0.0,1.0,0.0],
               [0.0,0.0,1.0],
               [1.0,0.0,1.0],
               [1.0,1.0,1.0],
               [0.0,1.0,1.0],
               ]
edge_list = [[1,2],[2,3],[3,4],[4,1],
             [5,6],[6,7],[7,8],[8,5],
             [1,5],[2,6],[3,7],[4,8]]
face_list = [[1,10,-5,-9],
             [2,11,-6,-10],
             [3,12,-7,-11],
             [4,9,-8,-12],
             [5,6,7,8],
             [-4,-3,-2,-1]]

body_list = [[1,2,3,4,5,6]]

initialize(vertex_list, edge_list, face_list,body_list)
print(len(global_state.VERTEXS))
energy = Area()
constraint = Volume(1.0)


# solver = ConjugateGradientSolver(M@M) #Use conjugate gradient solver if the matrix is too large

for i in range(3):
    iterate(energy, constraint, get_vertex_list1(),get_facet_list1(), num_iterations=100)
    refinement()
# 2648


########################################################################################################
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============= 替换为Matplotlib可视化 =============
vertices = np.array(get_vertex_list())
faces = np.array(get_facet_list())

# 创建图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 准备面数据
mesh_faces = []
for face in faces:
    if len(face) == 3:  # 三角形
        mesh_faces.append(vertices[face])
    elif len(face) == 4:  # 四边形分割为两个三角形
        mesh_faces.append(vertices[face[[0,1,2]]])
        mesh_faces.append(vertices[face[[0,2,3]]])

# 添加网格面片
mesh = Poly3DCollection(
    mesh_faces,
    alpha=0.8,
    linewidths=0.5,
    edgecolor='k',
    facecolor='lightblue'
)
ax.add_collection3d(mesh)

# 设置坐标轴
min_coord, max_coord = vertices.min(), vertices.max()
ax.set_xlim(min_coord, max_coord)
ax.set_ylim(min_coord, max_coord)
ax.set_zlim(min_coord, max_coord)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()