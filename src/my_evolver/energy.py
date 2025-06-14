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
    
# class ContactEnergy(Energy):
#     """A class to compute the contact energy of a liquid droplet on a solid surface."""
#     def __init__(self, contact_angle=90.0, surface_tension=1.0, plane_height=0.0):
#         """
#         Initialize the contact energy calculator.
        
#         Args:
#             contact_angle (float): Contact angle in degrees (typically between 0 and 180)
#             surface_tension (float): Liquid-gas surface tension coefficient
#             plane_height (float): Z-coordinate of the contact plane (default 0.0)
#         """
#         super().__init__('contact_energy')
#         self.theta = torch.deg2rad(torch.tensor(contact_angle))
#         self.sigma = surface_tension  # 表面张力系数
#         self.plane_height = plane_height  # 接触平面高度
#         self.plane_normal = torch.tensor([0., 0., 1.])  # 假设平面法向为Z轴
        
#     def compute_energy(self, Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the contact line energy based on Young's equation.
        
#         Args:
#             Verts: [N, 3] Tensor of vertex positions
#             Facets: [M, 3] Tensor of triangle vertex indices
            
#         Returns:
#             torch.Tensor: Scalar contact energy value
#         """
#         # 找出所有与平面相交的边
#         contact_edges = self._find_contact_edges(Verts, Facets)
        
#         # 计算接触线总长度
#         contact_length = torch.sum(contact_edges)
        
#         # 接触能: E = σ*(cosθ - 1)*L (Young方程)
#         return self.sigma * (torch.cos(self.theta) - 1) * contact_length
    
#     def _find_contact_edges(self, Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
#         """
#         Find all edges intersecting with the contact plane and compute their lengths.
        
#         Args:
#             Verts: [N, 3] Tensor of vertex positions
#             Facets: [M, 3] Tensor of triangle vertex indices
            
#         Returns:
#             torch.Tensor: Lengths of all contact edges
#         """
#         contact_lengths = []
        
#         for f in Facets:
#             # 获取三角形的三个顶点
#             v0, v1, v2 = Verts[f[0]], Verts[f[1]], Verts[f[2]]
            
#             # 检查三角形是否与平面相交
#             signs = torch.stack([
#                 v0[2] - self.plane_height,
#                 v1[2] - self.plane_height,
#                 v2[2] - self.plane_height
#             ])
            
#             # 如果符号不一致，说明三角形与平面相交
#             if not (torch.all(signs >= 0) or torch.all(signs <= 0)):
#                 # 找出与平面相交的两条边
#                 intersections = []
#                 for i in range(3):
#                     va = Verts[f[i]]
#                     vb = Verts[f[(i+1)%3]]
#                     if (va[2] - self.plane_height) * (vb[2] - self.plane_height) <= 0:
#                         # 计算交点
#                         t = (self.plane_height - va[2]) / (vb[2] - va[2])
#                         cross_point = va + t * (vb - va)
#                         intersections.append(cross_point)
                
#                 # 如果有两个交点，计算它们之间的距离
#                 if len(intersections) == 2:
#                     edge_length = torch.norm(intersections[1] - intersections[0])
#                     contact_lengths.append(edge_length)
        
#         return torch.stack(contact_lengths) if contact_lengths else torch.tensor(0.0)
    
#     def compute_and_store_gradient(self, Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the gradient of contact energy with respect to vertex positions.
        
#         Args:
#             Verts: [N, 3] Tensor of vertex positions
#             Facets: [M, 3] Tensor of triangle vertex indices
            
#         Returns:
#             torch.Tensor: [N, 3] gradient tensor
#         """
#         # 使用自动微分计算梯度
#         self.e_grad = torch.autograd.functional.jacobian(
#             lambda x: self.compute_energy(x, Facets), 
#             Verts,
#             create_graph=True
#         )
#         return self.e_grad

class ContactEnergy(Energy):
    def __init__(self, contact_angle=90.0, surface_tension=1.0, plane_height=0.0):
        super().__init__('contact_energy')
        self.theta = torch.deg2rad(torch.tensor(contact_angle))
        self.sigma = surface_tension
        self.plane_height = plane_height

    def compute_energy(self, Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
        contact_length = self._compute_contact_length(Verts, Facets)
        return self.sigma * (torch.cos(self.theta) - 1) * contact_length

    def _compute_contact_length(self, Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
        contact_length = 0.0
        edge_count = set()  # 用于去重
        
        for f in Facets:
            v0, v1, v2 = Verts[f[0]], Verts[f[1]], Verts[f[2]]
            # 统计在平面上的顶点数
            on_plane = [torch.isclose(v[2], torch.tensor(0.0)) for v in [v0, v1, v2]]
            num_on_plane = sum(on_plane)
            
            # 情况1：两个顶点在平面上（完全在平面上的边）
            if num_on_plane == 2:
                # 找到在平面上的两个顶点
                indices = [i for i, is_on in enumerate(on_plane) if is_on]
                va, vb = Verts[f[indices[0]]], Verts[f[indices[1]]]
                # 去重后计入长度
                edge_key = tuple(sorted((f[indices[0]].item(), f[indices[1]].item())))
                if edge_key not in edge_count:
                    contact_length += torch.norm(vb - va)
                    edge_count.add(edge_key)
        return contact_length

    def compute_and_store_gradient(self, Verts: torch.Tensor, Facets: torch.Tensor) -> torch.Tensor:
        self.e_grad = torch.autograd.functional.jacobian(
            lambda x: self.compute_energy(x, Facets), 
            Verts,
            create_graph=True
        )
        return self.e_grad