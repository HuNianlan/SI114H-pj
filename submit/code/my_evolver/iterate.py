
import torch
from optimizer import Adam,LineSearchSolver
from largesteps.solvers import CholeskySolver
import numpy as np

from web import webstruct

def iterate(web:webstruct,num_iterations:int=10):
    Verts = web.get_vertex_tensor()
    Faces = web.get_facet_tensor()
    Verts.requires_grad = True
    optimizer =Adam([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme
    # optimizer =LineSearchSolver(energy_fn)([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme
    
    # constraint = global_state.BODIES[0].constraints[0]
    for _ in (range(num_iterations)):
        mask = torch.tensor(web.get_vertex_mask())
        # 合并所有能量梯度
        E_grad = torch.zeros_like(Verts)
        for energy in web.ENERGY:
            E_grad += energy.compute_and_store_gradient(Verts, Faces)
        
        #Compute energy and volume gradients
        # E_grad = web.energy.compute_and_store_gradient(Verts,Faces)
        V_grad = torch.empty((len(web.BODIES),len(Verts),3)) #? 如果每个body只有一个constrains，后面要改
        target_val = torch.empty(len(web.BODIES))
        real_val = torch.empty(len(web.BODIES))
        
        for idx,b in enumerate(web.BODIES):
            for cons in b.constraints:
                b_f = Faces[np.array(b.get_facet_list())-web.facet_diff]
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

        Verts.grad=(E_grad-torch.sum(((V_grad.transpose(0,2))*F).transpose(0,2),dim=0))*mask.float()

        # Verts.grad = solver.solve(Verts.grad) # solver for linear system we can substitute with the cg solver

        #Gradient descent step
        # with torch.no_grad():
        #     Verts = Verts + V_grad*0.2
        # print(torch.max(Verts.grad))
        optimizer.step() 
        # web.project_boundary_points_to_circle(Verts)
        web.b_proj(Verts)


    web.update_vertex_coordinates(Verts)
