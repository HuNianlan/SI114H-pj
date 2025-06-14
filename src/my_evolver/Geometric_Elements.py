import torch
import global_state
from boundary import Boundary
class Vertex:
    """A class representing a vertex in a 3D space with an ID, coordinates, and neighbors.
    Each vertex can have multiple neighbors, which are also vertices."""
    _count:int = 0  # Class variable to keep track of the number of vertices
    def __init__(self, x, y,z=0,is_fixed:bool = False, boundary_func:Boundary = None):
        Vertex._count += 1
        self.vertex_id:int = Vertex._count # Unique ID for each vertex
        self.x:float = x
        self.y:float = y
        self.z:float = z
        self.coord = torch.tensor([x, y, z], dtype=torch.float32)  # Coordinates as a tensor
        self.is_fixed = is_fixed
        self.boundary_func = boundary_func
        if self.boundary_func == None:
            self.on_boundary = False
        else:
            self.on_boundary = True
        # self.E_grad:torch.Tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # Gradient placeholder
        # self.vgrad:torch.Tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # Volume gradient placeholder
    @classmethod
    def from_vertex_list(cls,v,is_fixed:bool = False, boundary_func:Boundary = None):
        """Create a Facet from three edges."""
        return cls(v[0], v[1], v[2],is_fixed,boundary_func)
    
    @classmethod
    def from_boundary_func(cls,par,is_fixed:bool = False, boundary_func:Boundary = None):
        """Create a Facet from three edges."""
        v = boundary_func.cal_cord(par)
        return cls(v[0], v[1], v[2],is_fixed,boundary_func)
    
    def __repr__(self):
        return f"Vertex(id={self.vertex_id}, x={self.x}, y={self.y}, z={self.z})"
    
    def move(self,x,y,z):
        if self.is_fixed:return
        self.x = x
        self.y = y
        self.z = z
        self.coord = torch.tensor([x, y, z], dtype=torch.float32)

    
        

class Edge:
    _count:int = 0  # Class variable to keep track of the number of edges
    """A class representing an edge in a 3D space, defined by two vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex,is_fixed = False,boundary_func:Boundary = None):
        Edge._count += 1
        self.edge_id:int = Edge._count  # Unique ID for each edge
        # self.edge_id:int = get_next_edge_id()
        # assert vertex1.vertex_id != vertex2.vertex_id, "Vertices must be different"
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
        self.is_fixed = is_fixed
        self.boundary_func = boundary_func
        if self.boundary_func == None:
            self.on_boundary = False
        else:
            self.on_boundary = True

    def __repr__(self):
        return f"Edge(vertex1={self.vertex1.vertex_id}, vertex2={self.vertex2.vertex_id})"
    def length(self):
        """Calculate the length of the edge."""
        return ((self.vertex1.x - self.vertex2.x) ** 2 + 
                (self.vertex1.y - self.vertex2.y) ** 2 + 
                (self.vertex1.z - self.vertex2.z) ** 2) ** 0.5
    def is_valid(self):
        return self.vertex1.vertex_id != self.vertex2.vertex_id


class Face:
    _count:int = 0  # Class variable to keep track of the number of faces
    """A class representing a face in a 3D space"""
    # def __init__(self, vertexs:list[Vertex]):
    #     Face._count += 1
    #     self.face_id:int = Face._count  # Unique ID for each face
    #     self.vertexs:list[Vertex] = vertexs
    
    def __init__(self, vertexs:list[Vertex],edges:list[Edge],ori:list[int],is_fixed = False,boundary_func = None):
        Face._count += 1
        self.face_id:int = Face._count  # Unique ID for each face
        self.edges:list[Edge] = edges
        self.ori:list[int] = ori
        self.vertexs:list[Vertex] = vertexs
        self.is_fixed = is_fixed
        self.boundary_func = boundary_func
        if self.boundary_func == None:
            self.on_boundary = False
        else:
            self.on_boundary = True
    
    def triangulation(self,facets:list=global_state.FACETS,vertexs:list = global_state.VERTEXS,edges:list = global_state.EDGES):
        """Triangulate the face by connecting each edge to the center point"""
        n = len(self.vertexs)
        if n == 3:
            facets.append(Facet(self.vertexs[0], self.vertexs[1], self.vertexs[2],self.face_id))  # Add triangle to the global list
            return
        center_x = sum(v.x for v in self.vertexs) / n
        center_y = sum(v.y for v in self.vertexs) / n
        center_z = sum(v.z for v in self.vertexs) / n

        center_vertex = Vertex(x=center_x, y=center_y, z=center_z)
        vertexs.append(center_vertex)  # Add center vertex to the global list
        for i in range(n):
            v1 = self.vertexs[i]
            v2 = self.vertexs[(i + 1) % n]
            e = Edge(center_vertex,v1)
            edges.append(e)
            facets.append(Facet(center_vertex, v1, v2,self.face_id))



class Facet:
    _count:int = 0  # Class variable to keep track of the number of facets
    """A class representing a face in a 3D space, defined by three vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex, vertex3:Vertex,face_id:int):
        # assert vertex1.vertex_id != vertex2.vertex_id and vertex1.vertex_id != vertex3.vertex_id and vertex2.vertex_id != vertex3.vertex_id, "Vertices must be different"
        Facet._count += 1
        self.facet_id:int = Facet._count  # Unique ID for each facet 
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
        self.vertex3:Vertex = vertex3
        self.vertex_idx:list[int] = [vertex1.vertex_id-1, vertex2.vertex_id-1, vertex3.vertex_id-1]  # List of vertex IDs
        self._face_id:int = face_id  # Placeholder for face ID, can be set later if needed
        self.volume:float = self.compute_volume()  # Placeholder for volume, can be calculated later
    @classmethod
    def from_edges(cls, edge1:Edge, edge2:Edge, edge3:Edge):
        """Create a Facet from three edges."""
        return cls(edge1.vertex1, edge2.vertex1, edge3.vertex1)
    
    def __repr__(self):
        return f"Face(vertex1={self.vertex1.vertex_id}, vertex2={self.vertex2.vertex_id}, vertex3={self.vertex3.vertex_id})"
    def area(self)-> float:
        """Calculate the area of the face using the cross product."""
        v1 = (self.vertex2.x - self.vertex1.x, 
               self.vertex2.y - self.vertex1.y, 
               self.vertex2.z - self.vertex1.z)
        v2 = (self.vertex3.x - self.vertex1.x, 
               self.vertex3.y - self.vertex1.y, 
               self.vertex3.z - self.vertex1.z)
        cross_product = (v1[1] * v2[2] - v1[2] * v2[1],
                         v1[2] * v2[0] - v1[0] * v2[2],
                         v1[0] * v2[1] - v1[1] * v2[0])
        return 0.5 * (cross_product[0]**2 + cross_product[1]**2 + cross_product[2]**2) ** 0.5
    
    def compute_volume(self) -> float:
        """Calculate the volume of the tetrahedron formed by this facet and the origin."""
        v1, v2, v3 = self.vertex1, self.vertex2, self.vertex3
        return abs((v1.x * (v2.y * v3.z - v3.y * v2.z) +
                     v2.x * (v3.y * v1.z - v1.y * v3.z) +
                     v3.x * (v1.y * v2.z - v2.y * v1.z)) / 6.0)

    def is_valid(self)->bool:
        return  self.vertex1.vertex_id != self.vertex2.vertex_id and self.vertex1.vertex_id != self.vertex3.vertex_id and self.vertex2.vertex_id != self.vertex3.vertex_id



from constraint import Constraint

class Body:
    _count:int = 0  # Class variable to keep track of the number of bodies
    """A class representing a 3D body composed of vertices, edges, and facets."""
    def __init__(self,face_list:list[int] = [],fixedvol:bool = False,volume:float = 0.0):
        """Initialize a Body with an optional list of facets."""
        Body._count += 1
        self.bid:int = Body._count  # Unique ID for each body
        self.directed_face_list:list[int]= face_list  # List of facet IDs that belong to this body
        self.faces:list[int] = []
        self.facets:list[int] = []  # List to store the IDs of the facets in this body
        self.face_sign:list[int] = []
        self.facet_sign:list[int] = []  # List to store the signs of the facets
        self.fixedvol:bool = fixedvol  # Whether the volume is fixed
        self.old_volume:float = volume  # Store the old volume for reference
        self.volume:float=volume
        self.constraints:list[Constraint] = []

    def __repr__(self):
        return f"Body(id={self.bid}, facets={len(self.facets)})"

    def add_facet_by_id(self,sign:int):
        """Add a facet to the body by its ID."""
        self.face_sign.append(sign)  # Store the sign of the facet
        self.faces.append(Face._count)  # Use the current count of facets to get the ID

    def add_constraints(self,constraint:Constraint):
        self.constraints.append(constraint)

    def compute_volume(self,FACETS:list[Facet] = global_state.FACETS) -> float:
        """Calculate the volume of the body using the divergence theorem."""
        volume = 0.0
        for facet in FACETS:
            if facet._face_id in self.faces:
                ind = self.faces.index(facet._face_id)
                sign = self.face_sign[ind]
                volume += sign * facet.volume
        self.volume = volume
        return volume


    def get_surface_area(self,FACETS:list[Facet] = global_state.FACETS) -> float:
        """Calculate the surface area of the body."""
        surface_area = 0.0
        for facet in FACETS:
            if facet._face_id in self.faces:
                surface_area += facet.area()
        return surface_area

    def update_facet_list(self,FACETS:list[Facet] = global_state.FACETS):
        """Update the list of facets in the body based on the current FACETS."""
        self.facets = [f.facet_id for f in FACETS if f._face_id in self.faces]

    def update_facet_sign(self,FACETS:list[Facet] = global_state.FACETS):
        """Get the signs of the facets in the body."""
        sign = []
        for f in FACETS:
            if f._face_id in self.faces:
                ind = self.faces.index(f._face_id)
                sign.append(self.face_sign[ind])
        self.facet_sign = sign

    def get_facet_list(self):
        return self.facets
    
    def get_facet_sign(self) -> torch.Tensor:
        return torch.tensor(self.facet_sign, dtype=torch.int8)
    
