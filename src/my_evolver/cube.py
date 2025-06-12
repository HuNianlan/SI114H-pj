# from utils import get_facet_list1,get_vertex_list1,get_facet_list,get_vertex_list
# from refinement import refinement
# from init import initialize
# from energy import Area,Energy
# from constraint import Volume,Constraint
from iterate import iterate
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
volume_constraint = [1.0]
########################################################################################################
# initialize(vertex_list, edge_list, face_list,body_list,volume_constraint)

# for i in range(3):
#     iterate(get_vertex_list1(),get_facet_list1(), num_iterations=5000)
#     refinement()
# 2648

from web import webstruct
web = webstruct(vertex_list, edge_list, face_list,body_list,volume_constraint)
for i in range(4):
    iterate(web, num_iterations=500)
    web.refinement()
########################################################################################################
from visualization import plot_mesh

plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")

