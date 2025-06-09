from utils import get_facet_list1,get_vertex_list1,get_facet_list,get_vertex_list
from refinement import refinement
from init import initialize
import polyscope as ps
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
import polyscope as ps
import numpy as np
ps.init()
ps.set_ground_plane_mode('none')
ps.register_surface_mesh("Mesh_result",np.array(get_vertex_list()),np.array(get_facet_list()))
# ps.register_surface_mesh("Mesh_result",Verts.detach().numpy(),Faces.numpy())
# ps.register_surface_mesh("Mesh_init",np.array(get_vertex_list()),np.array(get_facet_list()))
ps.show()
