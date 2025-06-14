from web import webstruct
from energy import Area, ContactEnergy
# from constraint import PlaneConstraint
from iterate import iterate
from boundary import LevelSetConstraint_Plane

boundary1 = LevelSetConstraint_Plane([0,0,1], [0,0,0])

vertex_list = [
    # 基底接触线顶点 (Z=0，constraint 1)
    [0.0, 0.0, 0.0, False, boundary1],   # 顶点1 (左下)
    [1.0, 0.0, 0.0, False, boundary1],    # 顶点2 (右下)
    [1.0, 1.0, 0.0, False, boundary1],    # 顶点3 (右上)
    [0.0, 1.0, 0.0, False, boundary1],    # 顶点4 (左上)
    
    # 液滴顶部顶点 (初始Z=1，可移动)
    [0.0, 0.0, 1.0, False, None],   # 顶点5 
    [1.0, 0.0, 1.0, False, None],   # 顶点6
    [1.0, 1.0, 1.0, False, None],   # 顶点7
    [0.0, 1.0, 1.0, False, None],   # 顶点8
    
    # 固定平面边界顶点 (fixed，用于可视化)
    [2.0, 2.0, 0.0, True, None],    # 顶点9
    [2.0, -1.0, 0.0, True, None],   # 顶点10
    [-1.0, -1.0, 0.0, True, None],  # 顶点11
    [-1.0, 2.0, 0.0, True, None]    # 顶点12
]

edge_list = [
    # ---- 基底接触线 (constraint 1) ----
    [1, 2, False, boundary1],   # 边1：顶点1 → 顶点2 (底部)
    [2, 3, False, boundary1],   # 边2：顶点2 → 顶点3 (右侧)
    [3, 4, False, boundary1],   # 边3：顶点3 → 顶点4 (顶部)
    [4, 1, False, boundary1],   # 边4：顶点4 → 顶点1 (左侧)
    
    # ---- 液滴顶部边 ----
    [5, 6, False, None],  # 边9：顶点5 → 顶点6
    [6, 7, False, None],  # 边10：顶点6 → 顶点7
    [7, 8, False, None],  # 边11：顶点7 → 顶点8
    [8, 5, False, None],  # 边12：顶点8 → 顶点5
    
    # ---- 液滴侧面边 ----
    [1, 5, False, None],  # 边5：顶点1 → 顶点5
    [2, 6, False, None],  # 边6：顶点2 → 顶点6
    [3, 7, False, None],  # 边7：顶点3 → 顶点7
    [4, 8, False, None],  # 边8：顶点4 → 顶点8
    
    # ---- 固定平面边界 (no_refine) ----
    [9, 10, True, None],  # 边13：顶点9 → 顶点10
    [10, 11, True, None], # 边14：顶点10 → 顶点11
    [11, 12, True, None], # 边15：顶点11 → 顶点12
    [12, 9, True, None]   # 边16：顶点12 → 顶点9
]

face_list = [
    [1, 10, -5,-9],
    [2, 11, -6, -10],
    [3, 12, -7, -11],
    [4, 9, -8, -12],
    [5, 6, 7, 8],
    [13, 14, 15, 16]
]

body_list = [[1, 2, 3, 4, 5]]
volume_constraint = [1.0]
energy_terms = [Area(), ContactEnergy()]

web = webstruct(vertex_list, edge_list, face_list,body_list,volume_constraint, energy_terms) 
for i in range(2):
    web.refinement()
    iterate(web, num_iterations=50)

########################################################################################################
from visualization import plot_mesh

plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")
