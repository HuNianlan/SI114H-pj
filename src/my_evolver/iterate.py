# 我们先构建一个模拟 "web" 的状态类，并实现相关依赖函数的框架。
import math
import random
from typing import List, Tuple
from web import web

gocount = 0  # 模拟迭代计数器

def save_current_coords():
    return [random.random() for _ in range(web.vertex_count)]

def restore_coords(saved):
    # 在真实实现中应将 saved 坐标赋值给当前系统状态
    pass

# 模拟梯度计算
def calc_all_gradients():
    web.total_energy *= 0.99 + 0.02 * random.random()  # 模拟能量变化

# 移动顶点
def move_vertices(test=True, scale=1.0):
    if test:
        web.total_energy *= 0.98 + 0.04 * random.random()  # 模拟测试移动能量
    else:
        web.total_energy *= 0.97 + 0.03 * random.random()  # 模拟真实移动能量

# 共轭梯度相关
def cg_calc_gamma(): pass
def cg_direction(): pass
def cg_restart(): pass

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

# # 测试 iterate 是否可以运行
# def iterate():
#     energy0 = web.total_energy
#     energy1 = energy2 = 0.0
#     scale0 = 0.0
#     scale1 = scale2 = 0.0
#     old_energy = web.total_energy
#     seek_count = 0

#     if web.vertex_count == 0:
#         print("No vertices. Did you forget to load a surface?")
#         return

#     if web.diffusion_flag:
#         diffuse()

#     if web.check_pinning_flag:
#         check_pinning()

#     if web.normal_motion_flag:
#         begin_normal_motion()

#     calc_all_gradients()

#     saved_coords = save_current_coords()

#     if web.conj_grad_flag:
#         cg_calc_gamma()
#         cg_direction()

#     if not web.motion_flag:
#         web.scale = web.scale if web.scale > 0 else web.max_scale * 1e-6
#         move_vertices(test=True, scale=web.scale)
#         energy1 = web.total_energy
#         scale1 = web.scale

#         restore_coords(saved_coords)
#         calc_all_gradients()
#         energy0 = web.total_energy

#         if energy1 < energy0:
#             while web.scale < web.max_scale:
#                 web.scale *= 2
#                 move_vertices(test=True, scale=web.scale)
#                 energy2 = web.total_energy
#                 scale2 = web.scale

#                 restore_coords(saved_coords)
#                 calc_all_gradients()

#                 if not math.isfinite(energy2) or energy2 > energy1:
#                     web.scale /= 2
#                     break
#                 energy1, scale1 = energy2, scale2
#         else:
#             while energy1 > energy0 and seek_count < 20:
#                 seek_count += 1
#                 energy2 = energy1
#                 scale2 = scale1
#                 web.scale /= 2

#                 if web.scale < 1e-12 * web.max_scale:
#                     web.scale = 0.0
#                     break

#                 move_vertices(test=True, scale=web.scale)
#                 energy1 = web.total_energy
#                 scale1 = web.scale

#                 restore_coords(saved_coords)
#                 calc_all_gradients()
#             web.scale *= 2

#         if web.scale > web.max_scale:
#             web.scale = web.max_scale
#         elif web.scale > 0.0:
#             denom = energy0 * (scale1 - scale2) + energy1 * (scale2 - scale0) + energy2 * (scale0 - scale1)
#             if denom != 0.0:
#                 web.scale = (
#                     (energy0 - energy2) * scale1**2 +
#                     (energy1 - energy0) * scale2**2 +
#                     (energy2 - energy1) * scale0**2
#                 ) / (2 * denom)
#     elif web.runge_kutta_flag:
#         runge_kutta()

#     if web.scale > web.max_scale:
#         web.scale = web.max_scale

#     move_vertices(test=False, scale=web.scale * web.scale_scale)

#     if web.jiggle_flag:
#         jiggle()

#     if web.autopop_flag or web.autochop_flag:
#         autopop_detect(web.scale)
#         if web.autopop_count or web.autochop_count:
#             autopop_pop()
#             autochop_chop()
#         autopop_cleanup()

#     if not math.isfinite(web.total_energy):
#         print("Motion would cause infinite energy. Restoring coordinates.")
#         restore_coords(saved_coords)
#         return

#     if web.check_increase_flag and web.total_energy > energy0:
#         print("Motion would have increased energy. Restoring coordinates.")
#         restore_coords(saved_coords)
#         return

#     if not web.motion_flag and web.total_energy > energy0:
#         restore_coords(saved_coords)
#         web.scale = scale1
#         move_vertices(test=False, scale=web.scale)
#         if web.cg_hvector:
#             cg_restart()

#     web.total_time += web.scale

#     if web.area_norm_flag and web.norm_check_flag and web.representation == 'SOAPFILM':
#         delta = normal_change_check()
#         if delta > web.norm_check_max:
#             print(f"Max normal change: {delta}. Restoring coordinates.")
#             restore_coords(saved_coords)
#             return
        
    
#     print(f"{gocount}, area:{web.total_area}, energy:{web.total_energy}, scale:{web.scale}")
# # 调用 iterate 一次看看效果
# iterate()

# def calc_all_grads():
#     # find_fixed()
#     calc_volgrads(DO_OPTS)
#     calc_force()
#     pressure_forces()
#     calc_lagrange()
#     lagrange_adjust()

import torch
from torch.optim import Adam as AdamUniform
from largesteps.solvers import CholeskySolver, ConjugateGradientSolver
from energy import Area,Energy
from constraint import Volume,Constraint
from Geometric_Elements import update_vertex_coordinates
from utils import get_facet_list1,get_vertex_list1


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

def iterate(energy:Energy, constraint:Constraint, Verts:torch.Tensor=get_vertex_list1(), Faces:torch.Tensor=get_facet_list1(),num_iterations:int=10):
    Verts.requires_grad = True
    optimizer =AdamUniform([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme
    lambda_=10.0
    M = compute_matrix(Verts, Faces, lambda_)
    solver = CholeskySolver(M@M)
    for i in (range(num_iterations)):
        #Compute energy and volume gradients
        E_grad = energy.compute_and_store_gradient(Verts,Faces)
        V_grad = constraint.compute_and_store_gradient(Verts,Faces)
        
        P = torch.sum(torch.sum(V_grad*V_grad,dim=1))
        Q = torch.sum(torch.sum(E_grad*V_grad,dim=1))
        R = (constraint.target_value-constraint.compute_constraint(Verts,Faces)) #residual volume
        F = Q/P
        M = R/P

        with torch.no_grad():
            Verts += V_grad*M

        Verts.grad=E_grad-F*V_grad
        Verts.grad = solver.solve(Verts.grad) # solver for linear system we can substitute with the cg solver

        #Gradient descent step
        optimizer.step() 

    update_vertex_coordinates(Verts)

