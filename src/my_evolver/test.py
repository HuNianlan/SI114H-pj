from Geometric_Elements import get_facet_list,get_vertex_list
import numpy as np
from refinement import refinement
from init import initialize

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

initialize(vertex_list, edge_list, face_list)

refinement()
refinement()











########################################################################################################
import polyscope as ps
ps.init()
ps.set_ground_plane_mode('none')
ps.register_surface_mesh("Mesh_result",np.array(get_vertex_list()),np.array(get_facet_list())-1)
ps.show()
