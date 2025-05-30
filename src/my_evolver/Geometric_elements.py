vertex_id = 0
def get_next_vertex_id():
    """Generate a unique vertex ID."""
    global vertex_id
    vertex_id += 1
    return vertex_id

edge_id = 0
def get_next_edge_id():
    """Generate a unique edge ID."""
    global edge_id
    edge_id += 1
    return edge_id


class Vertex:
    """A class representing a vertex in a 3D space with an ID, coordinates, and neighbors.
    Each vertex can have multiple neighbors, which are also vertices."""
    def __init__(self, x, y,z):
        self.id:int = get_next_vertex_id()
        self.x:float = x
        self.y:float = y
        self.z:float = z
    def __repr__(self):
        return f"Vertex(id={self.id}, x={self.x}, y={self.y}, z={self.z})"                              
    

class Edge:
    """A class representing an edge in a 3D space, defined by two vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex):
        self.edge_id:int = get_next_edge_id()
        assert vertex1.id != vertex2.id, "Vertices must be different"
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
    def __repr__(self):
        return f"Edge(vertex1={self.vertex1.id}, vertex2={self.vertex2.id})"
    def length(self):
        """Calculate the length of the edge."""
        return ((self.vertex1.x - self.vertex2.x) ** 2 + 
                (self.vertex1.y - self.vertex2.y) ** 2 + 
                (self.vertex1.z - self.vertex2.z) ** 2) ** 0.5
    
class Facet:
    """A class representing a face in a 3D space, defined by three vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex, vertex3:Vertex):
        assert vertex1.id != vertex2.id and vertex1.id != vertex3.id and vertex2.id != vertex3.id, "Vertices must be different"
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
        self.vertex3:Vertex = vertex3
    @classmethod
    def from_edges(cls, edge1:Edge, edge2:Edge, edge3:Edge):
        """Create a Facet from three edges."""
        return cls(edge1.vertex1, edge2.vertex1, edge3.vertex1)
    def __repr__(self):
        return f"Face(vertex1={self.vertex1.id}, vertex2={self.vertex2.id}, vertex3={self.vertex3.id})"
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


    

class Face:
    """A class representing a face in a 3D space"""
    def __init__(self, vertexs:list[Vertex]):
        self.vertexs:list[Vertex] = vertexs
    def triangulation(self):
        """Triangulate the face by connecting each edge to the center point"""
        n = len(self.vertexs)
        if n == 3:
            FACETS.append(Facet(self.vertexs[0], self.vertexs[1], self.vertexs[2]))  # Add triangle to the global list
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
            FACETS.append(Facet(center_vertex, v1, v2))
    
def faces_to_facets(faces:list[Face]):
    """Convert a list of Face objects to a list of Facet objects."""
    # print(len(VERTEXS))
    # print(len(EDGES))
    # print(len(FACETS))
    for face in faces:
        face.triangulation()
    # return facets




VERTEXS:list[Vertex] = []
EDGES:list[Edge] = []
FACETS:list[Facet] = []


def get_vertex_list() -> list[list[float]]:
    """Get the coordinates of all vertices."""
    return [[v.x, v.y, v.z] for v in VERTEXS]
def get_edge_list() -> list[list[int]]:
    """Get the list of edges as pairs of vertex IDs."""
    return [[e.vertex1.id, e.vertex2.id] for e in EDGES]
def get_facet_list() -> list[list[int]]:
    """Get the list of facets as triplets of vertex IDs."""
    return [[f.vertex1.id-1, f.vertex2.id-1, f.vertex3.id-1] for f in FACETS]


def create_vertices(vertex_list:list[list[float]]):
    """Create a list of Vertex objects from a list of coordinates and add it to VERTEXS."""
    for v in vertex_list:
        # VERTEXS.append(Vertex(v))
        VERTEXS.append(Vertex(x=v[0], y=v[1], z=v[2]))


def create_edges(edge_list:list[list[int]]):
    """Create a list of Edge objects from a list of vertex indices and add it to EDGES."""
    for e in edge_list:
        EDGES.append(Edge(vertex1=VERTEXS[e[0]-1], vertex2=VERTEXS[e[1]-1]))

def create_facets(face_list:list[list[int]]):
    """Create a list of Face objects from a list of edge indices."""
    faces:list[Face] = []
    for f in face_list:
        vertex_list = []
        for edge in f:
            if edge < 0:
                vertex_list.append(EDGES[-edge-1].vertex2)
            else:
                vertex_list.append(EDGES[edge-1].vertex1)
        # Create face edges from the edge indices
        faces.append(Face(vertexs=vertex_list))
    faces_to_facets(faces)
        # FACETS.extend(faces_to_facets(faces))  # Convert face to facets and add to FACETS

def find_vertex_by_coordinates(x:float, y:float, z:float) -> Vertex:
    """Find a vertex by its coordinates."""
    for v in VERTEXS:
        if v.x == x and v.y == y and v.z == z:
            return v
    return None

def find_edge_by_vertices(v1:Vertex, v2:Vertex) -> Edge:
    """Find an edge by its two vertices."""
    for e in EDGES:
        if (e.vertex1.id == v1.id and e.vertex2.id == v2.id) or (e.vertex1.id == v2.id and e.vertex2.id == v1.id):
            return e
    return None