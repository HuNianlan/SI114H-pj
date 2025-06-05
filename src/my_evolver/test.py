from Geometric_Elements import get_facet_list,get_vertex_list
import numpy as np
from refinement import refinement
from init import initialize
import polyscope as ps
from tqdm import tqdm
from utils import get_para, get_body_para
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

refinement()
refinement()


import torch
from torch.optim import Adam as AdamUniform
from largesteps.solvers import CholeskySolver

def compute_volume_manifold(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1],Coords[:,2],dim=1)
    determinants = torch.sum(cross_prods*Coords[:,0],dim=1)
    Vol = torch.sum(determinants)/6
    return(Vol)


def compute_area_manifold(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1]-Coords[:,0],Coords[:,2]-Coords[:,0],dim=1)
    Areas =  0.5*torch.norm(cross_prods,dim=1)
    return(torch.sum(Areas))


def compute_matrix(verts, faces, lambda_):
    """
    Build the parameterization matrix.
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    """
    L = laplacian_uniform(verts, faces)

    idx = torch.arange(verts.shape[0], dtype=torch.long, device=verts.device)
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device=verts.device), (verts.shape[0], verts.shape[0]))
    M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
    return M.coalesce()

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()



Verts = torch.tensor(get_vertex_list(), dtype=torch.float32)
Faces = torch.tensor(get_facet_list(), dtype=torch.int64)
Verts.requires_grad = True 
print("Volume of the mesh:", compute_volume_manifold(Verts, Faces).item())
print("Area of the mesh:", compute_area_manifold(Verts, Faces).item())

with torch.no_grad():
    Volume_target = compute_volume_manifold(Verts,Faces)#Equal to initial volume

optimizer =AdamUniform([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme
lambda_=10.0
M = compute_matrix(Verts, Faces, lambda_)
solver = CholeskySolver(M@M)

for i in (pbar:=tqdm(range(500))):
    #Compute energy and volume gradients
    E_grad = torch.autograd.functional.jacobian(lambda x: compute_area_manifold(x,Faces),Verts)
    V_grad = - torch.autograd.functional.jacobian(lambda x: compute_volume_manifold(x,Faces),Verts)
    P = torch.sum(torch.sum(V_grad*V_grad,dim=1))
    Q = torch.sum(torch.sum(E_grad*V_grad,dim=1))
    R = (Volume_target-compute_volume_manifold(Verts,Faces))
    F = Q/P
    M = R/P

    with torch.no_grad():
        Verts-= V_grad*M

    Verts.grad=E_grad-F*V_grad
    Verts.grad = solver.solve(Verts.grad)

    #Gradient descent step
    pbar.set_description("Area:"+str([compute_area_manifold(Verts,Faces).item()])+"  Volume:"+str(compute_volume_manifold(Verts,Faces).item()))
    optimizer.step() 



########################################################################################################
import polyscope as ps
ps.init()
ps.set_ground_plane_mode('none')
# ps.register_surface_mesh("Mesh_init",np.array(get_vertex_list()),np.array(get_facet_list()))
ps.register_surface_mesh("Mesh_result",Verts.detach().numpy(),Faces.numpy())
# ps.register_surface_mesh("Mesh_init",np.array(get_vertex_list()),np.array(get_facet_list()))
ps.show()
