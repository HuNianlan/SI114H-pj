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


VERTEXS:list = []
EDGES:list = []
FACETS:list = []



class Vertex:
    """A class representing a vertex in a 3D space with an ID, coordinates, and neighbors.
    Each vertex can have multiple neighbors, which are also vertices."""
    def __init__(self, x, y,z):
        self.id:int = get_next_vertex_id()
        self.x:float = x
        self.y:float = y
        self.z:float = z
        VERTEXS.append([x,y,z])
    def __repr__(self):
        return f"Vertex(id={self.id}, x={self.x}, y={self.y}, z={self.z})"                              
    

class Edge:
    """A class representing an edge in a 3D space, defined by two vertices."""
    def __init__(self, vertex1:Vertex, vertex2:Vertex):
        self.edge_id:int = get_next_edge_id()
        assert vertex1.id != vertex2.id, "Vertices must be different"
        self.vertex1:Vertex = vertex1
        self.vertex2:Vertex = vertex2
        EDGES.append([vertex1.id, vertex2.id])
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
        FACETS.append([vertex1.id, vertex2.id, vertex3.id])
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
    
    

class Face:
    """A class representing a face in a 3D space"""
    def __init__(self, vertexs:list[Vertex]):
        self.vertexs:list[Vertex] = vertexs
    def triangulation(self) -> list[Facet]:
        """Triangulate the face by connecting each edge to the center point"""
        n = len(self.vertexs)
        if n == 3:
            return [Facet(self.vertexs[0], self.vertexs[1], self.vertexs[2])]
        center_x = sum(v.x for v in self.vertexs) / n
        center_y = sum(v.y for v in self.vertexs) / n
        center_z = sum(v.z for v in self.vertexs) / n

        center_vertex = Vertex(x=center_x, y=center_y, z=center_z)

        triangles = []
        for i in range(n):
            v1 = self.vertexs[i]
            v2 = self.vertexs[(i + 1) % n]
            EDGES.append([center_vertex.id, v1.id])
            triangles.append(Facet(center_vertex, v1, v2))
        return triangles
    
def faces_to_facets(faces:list[Face]) -> list[Facet]:
    """Convert a list of Face objects to a list of Facet objects."""
    facets = []
    for face in faces:
        facets.extend(face.triangulation())
    return facets

