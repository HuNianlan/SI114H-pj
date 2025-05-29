from Geometric_elements import Vertex,Edge,Facet,Face,faces_to_facets
import numpy as np
from refinement import refinement
def create_vertices(vertex_list:list[list[float]]) -> list[Vertex]:
    """Create a list of Vertex objects from a list of coordinates."""
    return [Vertex(x=v[0], y=v[1], z=v[2]) for v in vertex_list]

def create_edges(edge_list:list[list[int]], vertices:list[Vertex]) -> list[Edge]:
    """Create a list of Edge objects from a list of vertex indices."""
    return [Edge(vertex1=vertices[e[0]-1], vertex2=vertices[e[1]-1]) for e in edge_list]

def create_faces(face_list:list[list[int]], edges:list[Edge]) -> list[Face]:
    """Create a list of Face objects from a list of edge indices."""
    faces = []
    for f in face_list:
        vertex_list = []
        for edge in f:
            if edge < 0:
                vertex_list.append(edges[-edge-1].vertex2)
            else:
                vertex_list.append(edges[edge-1].vertex1)
        # Create face edges from the edge indices
        faces.append(Face(vertexs=vertex_list))
    return faces




########################################################################################################
from Geometric_elements import FACETS,EDGES,VERTEXS

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

vertices = create_vertices(vertex_list)
edges = create_edges(edge_list, vertices)
faces = create_faces(face_list, edges)
facets = faces_to_facets(faces)


print(len(VERTEXS))
print(len(EDGES))
print(len(FACETS))
# Print the created geometric elements
# print("Vertices:")
# for v in vertices:
#     print(v)

# print("\nEdges:")
# for e in edges:
#     print(e)

# print("\nFacets:")
# for f in facets:
#     print(f)
refinement()

import polyscope as ps
ps.init()
ps.set_ground_plane_mode('none')
ps.register_surface_mesh("Mesh_result",np.array(VERTEXS),np.array(FACETS)-1)
ps.show()
