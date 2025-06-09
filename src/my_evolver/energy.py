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
        pass

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