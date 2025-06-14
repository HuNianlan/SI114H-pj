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

def compute_volume(Verts: torch.Tensor, Facets: torch.Tensor) -> float:
    """计算封闭曲面的体积（散度定理）可能在body中有这个函数，但是如何调用等下再调整"""
    volume = 0.0
    for f in Facets:
        v0, v1, v2 = Verts[f[0]], Verts[f[1]], Verts[f[2]]
        volume += torch.dot(v0, torch.cross(v1, v2)) / 6.0
    return volume.abs()  # 保证正值

class GravityPotential(Energy):
    """物理一致性如何处理？"""
    def __init__(self, gravity=torch.tensor([0.0, -9.8, 0.0]), density=1.0):
        super().__init__('gravity')
        self.gravity = gravity  
        self.density = density  # 质量密度（能量缩放因子）

    def compute_energy(self, Verts: torch.Tensor, Facets: torch.Tensor):
        # 计算质心高度（假设均匀密度）
        centroid_z = torch.mean(Verts[:, 2])  #? 所有顶点Z坐标均值
        # 计算当前体积（需提前通过Volume约束计算）
        volume = compute_volume(Verts, Facets)  #? 需要实现体积计算
        # 重力势能: E = density * (g·centroid) * volume
        return self.density * volume * torch.dot(self.gravity, 
                                              torch.tensor([0, 0, centroid_z]))
    
    def compute_and_store_gradient(self, Verts, Facets):
        # 梯度: F = -∇E = density * volume * g （均匀分布到所有顶点）
        volume = compute_volume(Verts, Facets)
        self.e_grad = torch.zeros_like(Verts)
        self.e_grad += (self.density * volume * self.gravity) / len(Verts)
        return self.e_grad
    
class ContactEnergy(Energy):
    def __init__(self, contact_angle=90, surface_tension=1.0):
        super().__init__('contact_energy')
        self.theta = torch.deg2rad(torch.tensor(contact_angle))
        self.sigma = surface_tension  # 表面张力系数
        self.plane_normal = torch.tensor([0., 0., 1.])  # 假设平面在Z=0

    def compute_energy(self, Verts: torch.Tensor, Facets: torch.Tensor):
        # 计算接触线长度（与平面相交的边）
        contact_length = 0.0
        for f in Facets:
            z_coords = Verts[f, 2]  # Z坐标
            if (z_coords.min() <= 0) != (z_coords.max() <= 0):  # 跨越Z=0
                contact_length += self._compute_edge_contact_length(Verts[f])
        
        # 接触能: E = σ*(cosθ - 1)*L (Young方程)
        return self.sigma * (torch.cos(self.theta) - 1) * contact_length

    def _compute_edge_contact_length(self, triangle_verts):
        # 计算三角形与平面的交线长度
        crossings = []
        for i in range(3):
            v1, v2 = triangle_verts[i], triangle_verts[(i+1)%3]
            if (v1[2] * v2[2]) <= 0:  # 边跨越Z=0
                t = -v1[2] / (v2[2] - v1[2])
                cross_point = v1 + t * (v2 - v1)
                crossings.append(cross_point[:2])  # 取XY平面投影
        if len(crossings) == 2:
            return torch.norm(crossings[0] - crossings[1])
        return 0.0