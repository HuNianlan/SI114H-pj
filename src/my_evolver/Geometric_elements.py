class Vertex:
    """A class representing a vertex in a 3D space with an ID, coordinates, and neighbors.
    Each vertex can have multiple neighbors, which are also vertices."""
    _count:int = 0  # Class variable to keep track of the number of vertices
    def __init__(self, x, y,z):
        Vertex._count += 1
        self.vertex_id:int = Vertex._count # Unique ID for each vertex
        # self.id:int = get_next_vertex_id()
        self.x:float = x
        self.y:float = y
        self.z:float = z
    def __repr__(self):
        return f"Vertex(id={self.vertex_id}, x={self.x}, y={self.y}, z={self.z})"


class Edge:
    _count:int = 0  # Class variable to keep track of the number of edges
    """A class representing an edge in a 3D space, defined by two vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex):
        Edge._count += 1
        self.edge_id:int = Edge._count  # Unique ID for each edge
        # self.edge_id:int = get_next_edge_id()
        assert vertex1.vertex_id != vertex2.vertex_id, "Vertices must be different"
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
    def __repr__(self):
        return f"Edge(vertex1={self.vertex1.vertex_id}, vertex2={self.vertex2.vertex_id})"
    def length(self):
        """Calculate the length of the edge."""
        return ((self.vertex1.x - self.vertex2.x) ** 2 + 
                (self.vertex1.y - self.vertex2.y) ** 2 + 
                (self.vertex1.z - self.vertex2.z) ** 2) ** 0.5

class Face:
    _count:int = 0  # Class variable to keep track of the number of faces
    """A class representing a face in a 3D space"""
    def __init__(self, vertexs:list[Vertex]):
        Face._count += 1
        self.face_id:int = Face._count  # Unique ID for each face
        self.vertexs:list[Vertex] = vertexs
    def triangulation(self):
        """Triangulate the face by connecting each edge to the center point"""
        n = len(self.vertexs)
        if n == 3:
            FACETS.append(Facet(self.vertexs[0], self.vertexs[1], self.vertexs[2],self.face_id))  # Add triangle to the global list
            return
        center_x = sum(v.x for v in self.vertexs) / n
        center_y = sum(v.y for v in self.vertexs) / n
        center_z = sum(v.z for v in self.vertexs) / n

        center_vertex = Vertex(x=center_x, y=center_y, z=center_z)
        VERTEXS.append(center_vertex)  # Add center vertex to the global list
        for i in range(n):
            v1 = self.vertexs[i]
            v2 = self.vertexs[(i + 1) % n]
            EDGES.append(Edge(center_vertex,v1))
            FACETS.append(Facet(center_vertex, v1, v2,self.face_id))



class Facet:
    _count:int = 0  # Class variable to keep track of the number of facets
    """A class representing a face in a 3D space, defined by three vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex, vertex3:Vertex,face_id:int):
        assert vertex1.vertex_id != vertex2.vertex_id and vertex1.vertex_id != vertex3.vertex_id and vertex2.vertex_id != vertex3.vertex_id, "Vertices must be different"
        Facet._count += 1
        self.facet_id:int = Facet._count  # Unique ID for each facet 
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
        self.vertex3:Vertex = vertex3
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
    # def self_refinement(self):
    #     """Refinement by creating new vertices at the midpoints of all edges and use these to 
    #     subdivide each facet into four new facets each similar to the original."""
    #     global VERTEXS, FACETS
    #     # Ensure the global lists are accessible
    #     v1 = self.vertex1
    #     v2 = self.vertex2
    #     v3 = self.vertex3
        
    #     # Calculate midpoints of edges
    #     mid12 = Vertex((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)
    #     mid23 = Vertex((v2.x + v3.x) / 2, (v2.y + v3.y) / 2, (v2.z + v3.z) / 2)
    #     mid31 = Vertex((v3.x + v1.x) / 2, (v3.y + v1.y) / 2, (v3.z + v1.z) / 2)
        
    #     VERTEXS.append(mid12)
    #     VERTEXS.append(mid23)   
    #     VERTEXS.append(mid31)
    #     # Create new facets
    #     FACETS.append(Facet(v1, mid12, mid31))
    #     FACETS.append(Facet(mid12, v2, mid23))
    #     FACETS.append(Facet(mid31, mid23, v3))
    #     FACETS.append(Facet(mid12, mid23, mid31))


    


def faces_to_facets(faces:list[Face]):
    """Convert a list of Face objects to a list of Facet objects."""
    # print(len(VERTEXS))
    # print(len(EDGES))
    # print(len(FACETS))
    for face in faces:
        face.triangulation()
    # return facets

class Body:
    _count:int = 0  # Class variable to keep track of the number of bodies
    """A class representing a 3D body composed of vertices, edges, and facets."""
    def __init__(self,face_list:list[int] = []):
        """Initialize a Body with an optional list of facets."""
        Body._count += 1
        self.bid:int = Body._count  # Unique ID for each body
        self.face_list:list[int]= face_list  # List of facet IDs that belong to this body
        self.directed_facets:list[int] = []
        self.volume:float=0.0

    def __repr__(self):
        return f"Body(id={self.bid}, facets={len(self.directed_facets)})"
    
    def add_facet_by_id(self,sign:int):
        """Add a facet to the body by its ID."""
        self.directed_facets.append(sign*Face._count)  # Use the current count of facets to get the ID
    def compute_volume(self) -> float:
        """Calculate the volume of the body using the divergence theorem."""
        volume = 0.0
        for facet in FACETS:
            if facet._face_id in self.directed_facets:
                volume += facet.volume
            elif -facet._face_id in self.directed_facets:
                volume -= facet.volume
        self.volume = volume
        return volume

    def get_surface_area(self) -> float:
        """Calculate the surface area of the body."""
        surface_area = 0.0
        for facet in FACETS:
            if facet._face_id in self.directed_facets or -facet._face_id in self.directed_facets:
                surface_area += facet.area()
        return surface_area

VERTEXS:list[Vertex] = []
EDGES:list[Edge] = []
FACES:list[Face] = []
FACETS:list[Facet] = []
BODIES:list[Body] = []

def get_vertex_list() -> list[list[float]]:
    """Get the coordinates of all vertices."""
    return [[v.x, v.y, v.z] for v in VERTEXS]
def get_edge_list() -> list[list[int]]:
    """Get the list of edges as pairs of vertex IDs."""
    return [[e.vertex1.vertex_id, e.vertex2.vertex_id] for e in EDGES]
def get_facet_list() -> list[list[int]]:
    """Get the list of facets as triplets of vertex IDs."""
    return [[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in FACETS]


def create_vertices(vertex_list:list[list[float]]):
    """Create a list of Vertex objects from a list of coordinates and add it to VERTEXS."""
    assert vertex_list[0].all()==0, "The first vertex must be at the origin (0, 0, 0)"
    for v in vertex_list:
        # VERTEXS.append(Vertex(v))
        VERTEXS.append(Vertex(x=v[0], y=v[1], z=v[2]))


def create_edges(edge_list:list[list[int]]):
    """Create a list of Edge objects from a list of vertex indices and add it to EDGES."""
    for e in edge_list:
        EDGES.append(Edge(vertex1=VERTEXS[e[0]-1], vertex2=VERTEXS[e[1]-1]))

def create_facets(face_list:list[list[int]]):
    for f in face_list:
        vertex_list = []
        for edge in f:
            if edge < 0:
                vertex_list.append(EDGES[-edge-1].vertex2)
            else:
                vertex_list.append(EDGES[edge-1].vertex1)
        # Create face edges from the edge indices
        f = Face(vertexs=vertex_list)
        FACES.append(f)
        for body in BODIES:
            for fid in body.face_list:
                if abs(fid) == f.face_id:
                    body.add_facet_by_id(sign=1 if fid > 0 else -1)
                    break
        f.triangulation()
    

# def create_facets():

        # FACETS.extend(faces_to_facets(faces))  # Convert face to facets and add to FACETS

def create_bodies(body_list:list[list[int]]):
    """Create a list of Body objects from a list of facet indices."""
    for b in body_list:
        BODIES.append(Body(face_list=b))  # Initialize with empty faces

def find_vertex_by_coordinates(x:float, y:float, z:float) -> Vertex:
    """Find a vertex by its coordinates."""
    for v in VERTEXS:
        if v.x == x and v.y == y and v.z == z:
            return v
    return None

def find_edge_by_vertices(v1:Vertex, v2:Vertex) -> Edge:
    """Find an edge by its two vertices."""
    for e in EDGES:
        if (e.vertex1.vertex_id == v1.vertex_id and e.vertex2.vertex_id == v2.vertex_id) or (e.vertex1.vertex_id == v2.vertex_id and e.vertex2.vertex_id == v1.vertex_id):
            return e
    return None
