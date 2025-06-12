# from Geometric_Elements import Vertex, Facet, Edge, Body
# from utils import get_vertex_list1,get_facet_list1
import torch
class Energy:
    '''A class to compute the energy of a globalgeometric body based on its vertices, facets.'''
    def __init__(self,name:str):
        self.name:str = name
        self.e_grad:torch.Tensor = None
    def compute_energy(self,Verts:torch.Tensor,Facets:torch.Tensor):
        # Placeholder for energy computation logic
        pass
    def compute_and_store_gradient(self,Verts:torch.Tensor,Facets:torch.Tensor):
        '''Compute the gradient of the energy with respect to the vertices.'''
        # Placeholder for gradient computation logic
        self.e_grad=torch.autograd.functional.jacobian(lambda x: self.compute_energy(x,Facets),Verts)
        # pass

class Area(Energy):
    '''A class to compute the area energy of a global geometric body.'''
    def __init__(self):
        super().__init__('area')

    def compute_energy(self,Verts:torch.Tensor,Facets:torch.Tensor):
        '''Compute the area energy of the body.'''
        Coords = Verts[Facets]
        cross_prods = torch.cross(Coords[:,1]-Coords[:,0],Coords[:,2]-Coords[:,0],dim=1)
        Areas =  0.5*torch.norm(cross_prods,dim=1)
        return(torch.sum(Areas))
    

    def compute_and_store_gradient(self,Verts:torch.Tensor,Facets:torch.Tensor):
        '''Compute the gradient of the area energy with respect to the vertices.'''
        self.e_grad=torch.autograd.functional.jacobian(lambda x: self.compute_energy(x,Facets),Verts)
        return self.e_grad
    

class Sq_Mean_Curvature(Energy):
    def __init__(self):
        super().__init__('sq_mean_curvature')

    # def compute_area(self,Verts:torch.Tensor,Facets:torch.Tensor):


    def compute_energy(self,Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
        """
        Compute the integral of squared mean curvature energy over the mesh
        using the method derived from surface tension forces.

        Verts: [N, 3] Tensor of vertex positions
        Facets: [M, 3] Tensor of triangle vertex indices
        Returns: scalar Tensor energy
        """
        N = Verts.shape[0]
        M = Facets.shape[0]

        # 每个顶点的 area, force, normal
        device = Verts.device
        area = torch.zeros(N, device=device)
        force = torch.zeros((N, 3), device=device)
        normal = torch.zeros((N, 3), device=device)

        for f in Facets:
            v0, v1, v2 = f.tolist()
            p0, p1, p2 = Verts[v0], Verts[v1], Verts[v2]

            # 边向量
            t1 = p1 - p0  # side[0]
            t2 = p2 - p0  # side[1]

            # 点积
            t1t1 = torch.dot(t1, t1)
            t1t2 = torch.dot(t1, t2)
            t2t2 = torch.dot(t2, t2)

            # 面积
            det = t1t1 * t2t2 - t1t2 * t1t2
            tri_area = torch.sqrt(torch.clamp(det, min=1e-12)) / 2.0

            # 累加到每个顶点的面积
            for vi in [v0, v1, v2]:
                area[vi] += tri_area

            # 力贡献
            if tri_area > 0:
                coeff1 = (t2t2 * t1 - t1t2 * t2) / (4 * tri_area)
                coeff2 = (t1t1 * t2 - t1t2 * t1) / (4 * tri_area)

                force[v0] -= coeff1
                force[v1] += coeff1

                force[v2] += coeff2
                force[v1] -= coeff2

            # 法向量贡献（可选）
            n = torch.cross(t1, t2)
            for vi in [v0, v1, v2]:
                normal[vi] += n

        # 计算每个顶点的能量：E_v = (3/4) * ||F||^2 / A_v
        force_squared = (force ** 2).sum(dim=1)
        per_vertex_energy = (3.0 / 4.0) * force_squared / area.clamp(min=1e-8)
        total_energy = per_vertex_energy.sum()

        return total_energy


    def compute_and_store_gradient(self,Verts:torch.Tensor,Facets:torch.Tensor):
        '''Compute the gradient of the area energy with respect to the vertices.'''
        self.e_grad=torch.autograd.functional.jacobian(lambda x: self.compute_energy(x,Facets),Verts)
        return self.e_grad

# The integral of squared mean curvature in the soapfilm model is calculated for this method as follows: Each vertex v has a star of facets around it of area A.
# The force F due to surface tension on the vertex is the gradient of area, Since each facet has 3 vertices, the area associated with v is A/3. Hence the average mean curvature at v is

# h = (1/2)(F/(A/3)),
# where the 1/2 factor comes from the "mean" part of "mean curvature". This vertex's contribution to the total integral is then
# E = h2A/3 = (3/4)F^2/A.


class GravityPotential(Energy):
    def __init__(self, gravity=torch.tensor([0.0, -9.8, 0.0])):
        super().__init__('gravity')
        self.gravity = gravity  # 重力加速度向量

    def compute_energy(self, Verts: torch.Tensor, Facets: torch.Tensor):
        # 重力势能: E = -m * g * y (假设质量为1)
        return -torch.sum(Verts[:, 0.000000001] * self.gravity[1])  # y方向分量

    def compute_and_store_gradient(self, Verts, Facets):
        # 梯度: F = -∇E = m * g (方向向下)
        self.e_grad = torch.zeros_like(Verts)
        self.e_grad[:, :3] = self.gravity  # 所有顶点受相同的力
        return self.e_grad