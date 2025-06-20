import torch
class Constraint:
    '''A class to represent a constraint on the geometric body.'''
    def __init__(self, name: str,target_value: float = 0.0):
        self.name: str = name
        self.c_grad: torch.Tensor = None
        self.target_value: float = target_value  # Target value for the constraint

    def compute_constraint(self, Verts: torch.Tensor, Facets: torch.Tensor):
        # Placeholder for constraint computation logic
        pass
    def compute_and_store_gradient(self, Verts: torch.Tensor, Facets: torch.Tensor)->torch.Tensor:
        '''Compute the gradient of the constraint with respect to the vertices.'''
        # Placeholder for gradient computation logic
        pass

class Volume(Constraint):
    '''A class to compute the volume constraint of a global geometric body.'''
    def __init__(self,target_volume: float = 0.0):
        super().__init__('volume', target_value=target_volume)

    def compute_constraint(self, Verts: torch.Tensor, Facets: torch.Tensor, Signs: torch.Tensor=None):
        '''Compute the oriented volume constraint of the body.'''
        # print(Facets.shape)
        Coords = Verts[Facets]           # [F, 3, 3]
        v0, v1, v2 = Coords[:, 0], Coords[:, 1], Coords[:, 2]
        volume_per_face = torch.sum(v0 * torch.cross(v1, v2, dim=1), dim=1)  # [F]
        if Signs is None:
            Signs = torch.ones_like(volume_per_face)
        signed_volume = torch.sum(Signs * volume_per_face) / 6
        return signed_volume

    def compute_and_store_gradient(self,Verts:torch.Tensor,Facets:torch.Tensor, Signs: torch.Tensor=None):
        '''Compute the gradient of the volume constraint with respect to the vertices.'''
        self.e_grad=torch.autograd.functional.jacobian(lambda x: self.compute_constraint(x,Facets,Signs),Verts)
        return self.e_grad