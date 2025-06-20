from iterate import iterate

vertices = [
    [0.0, 0.0, 0.0],  # 1
    [1.0, 0.0, 0.0],  # 2
    [1.0, 1.0, 0.0],  # 3
    [0.0, 1.0, 0.0],  # 4
    [0.0, 0.0, 1.0],  # 5
    [1.0, 0.0, 1.0],  # 6
    [1.0, 1.0, 1.0],  # 7
    [0.0, 1.0, 1.0],  # 8
    [1.0, 2.0, 0.0],  # 9
    [0.0, 2.0, 0.0],  # 10
    [0.0, 2.0, 1.0],  # 11
    [1.0, 2.0, 1.0],  # 12
]

edges = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 1],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 5],
    [1, 5],
    [2, 6],
    [3, 7],
    [4, 8],
    [3, 9],
    [4, 10],
    [8, 11],
    [7, 12],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 9],
]

faces = [
    [1, 10, -5, -9],
    [2, 11, -6, -10],
    [3, 12, -7, -11],
    [4, 9, -8, -12],
    [5, 6, 7, 8],
    [-4, -3, -2, -1],
    [13, 17, -14, -3],
    [13, -20, -16, -11],
    [17, 18, 19, 20],
    [14, 18, -15, -12],
    [16, -19, -15, -7],
]

bodies = [[1, 2, 3, 4, 5, 6], [-3, -7, 8, 9, -10, 11]]
volume_constraint = [1.0,2.0]
##################################################################
from web import webstruct
web = webstruct(vertices, edges, faces, bodies,volume_constraint)
for i in range(3):
    iterate(web, num_iterations=50)
    web.refinement()

# from energy import Area
# import math
# print(f"facets:{len(web.FACETS)}")
# print(f"max angle: {math.degrees(web.compute_max_normal_angle())}")
# print(f"energy: {Area().compute_energy(web.get_vertex_tensor(), web.get_facet_tensor())}")
##################################################################
from visualization import plot_mesh
plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")