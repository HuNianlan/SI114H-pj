# 我们先构建一个模拟 "web" 的状态类，并实现相关依赖函数的框架。
import math
import random
from typing import List, Tuple
import global_state
gocount = 0  # 模拟迭代计数器

# Diffusion / pinning 等
def diffuse(): pass
def check_pinning(): pass
def begin_normal_motion(): pass

# Runge-Kutta
def runge_kutta(): pass

# 其他操作
def jiggle(): pass
def autopop_detect(scale): pass
def autopop_pop(): pass
def autochop_chop(): pass
def autopop_cleanup(): pass

# Normal check & energy估算
def normal_change_check() -> float:
    return random.uniform(0, 2.0)

def estimate_decrease() -> float:
    return -random.uniform(0.1, 1.0)


# def calc_all_grads():
#     # find_fixed()
#     calc_volgrads(DO_OPTS)
#     calc_force()
#     pressure_forces()
#     calc_lagrange()
#     lagrange_adjust()

import torch
from optimizer import Adam,LineSearchSolver
from largesteps.solvers import CholeskySolver, ConjugateGradientSolver
from energy import Area,Energy
from constraint import Volume,Constraint
from Geometric_Elements import update_vertex_coordinates
from utils import get_facet_list1,get_vertex_list1,get_body_list1


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

import global_state
import numpy as np
def iterate(Verts:torch.Tensor=get_vertex_list1(), Faces:torch.Tensor=get_facet_list1(),energy:Energy=Area(),num_iterations:int=10):
    Verts.requires_grad = True
    optimizer =Adam([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme
    # optimizer =LineSearchSolver(energy_fn)([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme

    lambda_=10.0
    M = compute_matrix(Verts, Faces, lambda_)
    solver = CholeskySolver(M@M)
    # constraint = global_state.BODIES[0].constraints[0]
    for _ in (range(num_iterations)):
        #Compute energy and volume gradients
        E_grad = energy.compute_and_store_gradient(Verts,Faces)
        V_grad = torch.empty((len(global_state.BODIES),len(Verts),3)) # 如果每个body只有一个constrains，后面要改
        target_val = torch.empty(len(global_state.BODIES))
        real_val = torch.empty(len(global_state.BODIES))
        
        for idx,b in enumerate(global_state.BODIES):
            for cons in b.constraints:
                b_f = Faces[np.array(b.get_facet_list())-global_state.facet_diff]
                v_grad = cons.compute_and_store_gradient(Verts,b_f,Signs = b.get_facet_sign())
                V_grad[idx]= -v_grad
                target_val[idx]=cons.target_value
                real_val[idx] = cons.compute_constraint(Verts,b_f,Signs = b.get_facet_sign())

        P = torch.sum(torch.sum((V_grad*V_grad.unsqueeze(1)),dim=3),dim=2)
        Q = torch.sum(torch.sum(E_grad*V_grad,dim=2),dim=1)
        
        R = (target_val-real_val) #residual volume
        F = torch.linalg.solve(P,Q)
        M = torch.linalg.solve(P,R)

        with torch.no_grad():
            Verts -= torch.sum(((V_grad.transpose(0,2))*M).transpose(0,2),dim=0)

        Verts.grad=E_grad-torch.sum(((V_grad.transpose(0,2))*F).transpose(0,2),dim=0)

        Verts.grad = solver.solve(Verts.grad) # solver for linear system we can substitute with the cg solver

        #Gradient descent step
        # with torch.no_grad():
        #     Verts = Verts + V_grad*0.2
        optimizer.step() 

    update_vertex_coordinates(Verts)

def iterate_catenoid(Verts: torch.Tensor = get_vertex_list1(), Faces: torch.Tensor = get_facet_list1(), energy: Energy = Area(), num_iterations: int = 10): 
    Verts.requires_grad = True
    optimizer = Adam([{'params': Verts, 'lr': 0.01}])

    lambda_ = 10.0
    M = compute_matrix(Verts, Faces, lambda_)
    solver = CholeskySolver(M @ M)

    fixed_mask = torch.tensor([v.is_fixed for v in global_state.VERTEXS], dtype=torch.bool).to(Verts.device)  # shape: (V,)
    
    for _ in range(num_iterations):
        # 1. 计算面积能量的梯度
        E_grad = energy.compute_and_store_gradient(Verts, Faces)

        # 4. 设置 Verts 的梯度
        Verts.grad = -E_grad

        # 5. 梯度下降更新
        optimizer.step()

    # 7. 更新 global_state 中的顶点坐标（坐标写回 Vertex 对象

    update_vertex_coordinates(Verts)
