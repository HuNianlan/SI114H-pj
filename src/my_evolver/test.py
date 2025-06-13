# from utils import get_facet_list1,get_vertex_list1,get_facet_list,get_vertex_list
# from refinement import refinement
# from init import initialize
# from energy import Area,Energy
# from constraint import Volume,Constraint
from iterate import iterate
########################################################################################################
from math import pi,cos,sin
from boundary import Boundary,Ring_XZ
RMAX = 0.1   # minimum radius for height
ZMAX = 1.0


def bound_func1(rho):#upper ring
    x = RMAX *cos(rho[0])+0.5
    z = RMAX *sin(rho[0])+0.5
    y = 1
    return [x,y,z]

def bound_func2(rho):#lower ring
    x = RMAX *cos(rho[0])+0.5
    z = RMAX *sin(rho[0])+0.5
    y = 0
    return [x,y,z]

boundary1 = Ring_XZ([0.5,0.5],RMAX,None)
boundary2 = Ring_XZ([0.5,0.5],RMAX,None)

vertex_list = [
    [0.0, 0.0, 0.0],  # 1
    [1.0, 0.0, 0.0],  # 2
    [1.0, 1.0, 0.0],  # 3
    [0.0, 1.0, 0.0],  # 4
    [0.0, 0.0, 1.0],  # 5
    [1.0, 0.0, 1.0],  # 6
    [1.0, 1.0, 1.0],  # 7
    [0.0, 1.0, 1.0],  # 8

    [[0.25, 0, 0.25],False,boundary2],  # 9
    [[0.75, 0, 0.25],False,boundary2],  # 10
    [[0.75, 1, 0.25],False,boundary1],  # 11
    [[0.25, 1, 0.25],False,boundary1],  # 12
    [[0.25, 0, 0.75],False,boundary2],  # 13
    [[0.75, 0, 0.75],False,boundary2],  # 14
    [[0.75, 1, 0.75],False,boundary1],  # 15
    [[0.25, 1, 0.75],False,boundary1],  # 16
]

edge_list = [
    [1, 2],  [2, 3],  [3, 4],  [4, 1],      # 1–4: 底部
    [5, 6],  [6, 7],  [7, 8],  [8, 5],      # 5–8: 顶部
    [1, 5],  [2, 6],  [3, 7],  [4, 8],      # 9–12: 立柱
    
    [9,10,False,boundary2], [10,11], [11,12,False,boundary1], [12,9],       # 13–16: 小立方体底
    [13,14,False,boundary2], [14,15], [15,16,False,boundary1], [16,13],     # 17–20: 小立方体顶
    
    [9,13,False,boundary2], [10,14,False,boundary2], [11,15,False,boundary1], [12,16,False,boundary1],      # 21–24: 小立方体柱
    
    [1,9], [2,10], [3,11], [4,12],          # 37–40: 外后→内后
    [5,13], [6,14], [7,15], [8,16]        # 29–32: 外前→内前
]

face_list = [
    [-4,-3,-2,-1],      # 1: 底面
    [2, 11, -6, -10],     # 2: 右面
    [5,6,7,8],     # 3: 顶面
    [4, 9, -8, -12],      # 4: 左面

    # 原来的前面（面 5）分成：
    [25,21,-29,-9],     # 5: 前面-左梯形
    [29,17, -30, -5],    # 6: 前面-上梯形
    [10,30,-22,-26],    # 7: 前面-右梯形
    [1, 26, -13, -25],    # 8: 前面-下梯形

    #原来的后面（面 6）分成：
    [3, 28, -15, -27],     # 9: 后面-下梯形
    [12, 32, -24, -28],    # 10: 后面-右梯形
    [31,19,-32,-7],    # 11: 后面-上梯形
    [27,23,-31,-11],    # 12: 后面-左梯形

    #小立方体内腔（朝内法向）
    # [13,14,15,16],
    [-16,-15,-14,-13], # 13: 底
    [14, 23, -18, -22], # 14: 右
    [17, 18, 19, 20], # 15: 顶
    [16, 21, -20, -24], # 16: 左
    # [24,20,-21,-16]
    #前后两个面不显示
    # [13,22,-17,-21],
    # [15,24,-19,-23]

    # #新增面：
    # [29,-20,-32,8],#17
    # [25,-16,-28,4],#18
    # [26,14,-27,-2],#19
    # [-30,6,31,-18]#20
]


body_list = [
    [1, 2, 3, 4,
    5, 6, 7,8, 
    9,10, 11, 12,
    -13, -14, -15, -16, 
    # 17,18
    # 17, 18,19,20
    ],                 # 小立方体中空体
]
# body_list = [[-13, 14, 15,-16,17,18]]
volume_constraint = [0.75]



########################################################################################################
# initialize(vertex_list, edge_list, face_list,body_list,volume_constraint)

# for i in range(3):
#     iterate(get_vertex_list1(),get_facet_list1(), num_iterations=5000)
#     refinement()
# 2648

from web import webstruct
from energy import Sq_Mean_Curvature




# web = webstruct(vertex_list, edge_list, face_list,body_list,volume_constraint,Sq_Mean_Curvature())
web = webstruct(vertex_list, edge_list, face_list,body_list,volume_constraint)
for i in range(3):
    web.refinement()
    iterate(web, num_iterations=50)

########################################################################################################
from visualization import plot_mesh

plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")

# plot_mesh(np.array(vertex_list), (np.abs(np.array(face_list))-1), "Optimized Mesh")
