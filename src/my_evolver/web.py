# struct webstruct {
#      struct skeleton skel[NUMELEMENTS];
#      int sizes[NUMELEMENTS];  /* allocated space for element structure */
#      int usedsizes[NUMELEMENTS];  /* used space for element structure */
#      struct element **elhashtable; /* id hash list of element pointers */
#      int elhashcount;  /* actual number of live entries */
#      int elhashmask;  /* for picking off index bits of id hash */
#      int elhashsize;  /* size of hash table; power of 2 */
#      int sdim;  /* dimension of ambient space */
#      int dimension;    /* where tension resides */
#      int representation; /* STRING, SOAPFILM, or SIMPLEX */
#      int modeltype;    /* QUADRATIC, LINEAR, or LAGRANGE; see defines below */
#      int lagrange_order; /* polynomial order of elements */
#      int headvnum;  /* number of head vertex in edge list */
#      int maxparam;    /* maximum number of parameters in any boundary */
#      int maxcon;      /* number of constraint structures allocated */
#      int highcon;     /* highest constraint number used */
#      struct constraint  *constraints; /* constraint definitions */
#      conmap_t con_global_map[MAXCONPER]; /* global vertex constraints */
#      int con_global_count;  /* number of global vertex constraints */
#      REAL tolerance;      /* constraint error tolerance */
#      REAL target_tolerance; /* error tolerance for extensive constraints */
#      int bdrymax;        /* number of boundary structures allocated */
#      int highbdry;       /* highest boundary number used */
#      struct boundary *boundaries; /* for free boundaries */
#      int diffusion_flag;  /* whether diffusion in effect */
#      REAL diffusion_const;  /* coefficient for diffusion */
#      REAL simplex_factorial; /* content correction factor for determinant */
#      int torus_clip_flag;
#      int torus_body_flag;
#      int symmetric_content; /* 1 if volumes use symmetric divergence */
#      int h_inverse_metric_flag; /* for laplacian of curvature */
#      REAL meritfactor;    /* for multiplying figure of merit */
#      int gravflag;         /* whether gravity is on */
#      REAL grav_const;      /* multiplier for gravitational force */
#      int convex_flag;     /* whether any convex boundaries present */
#      int pressflag;        /* whether prescribed pressures present */
#      int constr_flag;     /* set if there are any one-sided constraints */
#      int motion_flag;     /* set for fixed scale of motion;
#                                           otherwise seek minimum. */
#      int symmetry_flag;  /* whether symmetry group in effect */
#      int torus_flag;     /* whether working in toroidal domain */
#      int full_flag;     /* whether torus solidly packed with bodies */
#      int pressure_flag;  /* whether pressure used dynamically */
#      int projection_flag; /* whether to project */
#      int area_norm_flag; /* whether to normalize force by area surrounding vertex */
#      int norm_check_flag;  /* whether area normalization checks normal deviation */
#      REAL norm_check_max;  /* maximum allowable deviation */
#      int vol_flag;         /* whether body volumes up to date */
#      int jiggle_flag;     /* whether to jiggle vertices at each move */
#      int homothety;        /* flag for homothety adjustment each iteration */
#      int wulff_flag;      /* whether we are using wulff shapes for energy */
#      int wulff_count;     /* number of Wulff vectors read in */
#      char wulff_name[60]; /* Wulff file or keyword */
#      vertex_id  zoom_v;    /* vertex to zoom on */
#      REAL zoom_radius;     /* current zoom radius */
#      REAL total_area;
#      REAL total_area_addends[MAXADDENDS]; /* for binary tree addition */
#      REAL total_energy;
#      REAL total_energy_addends[MAXADDENDS]; /* for binary tree addition */
#      REAL spring_energy;
#      int total_facets;
#      int bodycount;  /* number of bodies */
#      body_id outside_body;  /* a body surrounding all others */
#      REAL scale;     /* force to motion scale factor */
#      REAL scale_scale;     /* over-relaxation factor */
#      REAL maxscale;     /* upper limit on scale factor */
#      REAL pressure;    /* ambient pressure */
#      REAL min_area;        /* criterion on weeding out small triangles */
#      REAL min_length;     /* criterion on weeding out small triangles */
#      REAL max_len;         /* criterion for dividing long edges */
#      REAL max_angle;      /* max allowed deviation from parallelism */
#      REAL temperature;  /* "temperature" for jiggling */
#      REAL spring_constant;  /* for forcing edges to conform to boundary */
#      int  gauss1D_order;        /* order for gaussian 1D integration */
#      int  gauss2D_order;        /* order for gaussian 2D integration */
#      REAL torusv;                 /* unit cell volume or area */
#      REAL **torus_period;
#      REAL **inverse_periods;/* inverse matrix of torus periods */
#      REAL **inverse_periods_tr;/* transpose of inverse matrix of periods */
#      REAL **torus_display_period;
#      REAL display_origin[MAXCOORD];
#      REAL **inverse_display_periods;/* inverse of torus display periods */
#      int  metric_flag;      /* set if background metric in force */
#      int  conformal_flag;  /* set for conformal metrics */
#      struct expnode metric[MAXCOORD][MAXCOORD]; /* metric component functions */
      
#      /* Some counters.  New scheme: Using word of flag bits for having-been-
#         reported status and needing-reported status, so exec() doesn't have
#         to zero these for each call to exec. */
#      /* Counts that are the result of mass action only are reported 
#         immediately */
#      int equi_count; 
#      int edge_delete_count; 
#      int facet_delete_count; 
#      int edge_refine_count; 
#      int facet_refine_count; 
#      int vertex_dissolve_count; 
#      int edge_dissolve_count; 
#      int facet_dissolve_count; 
#      int body_dissolve_count; 
#      int edge_reverse_count;
#      int facet_reverse_count;
#      int vertex_pop_count; 
#      int edge_pop_count; 
#      int pop_tri_to_edge_count;
#      int pop_edge_to_tri_count;
#      int pop_quad_to_quad_count;
#      int where_count; 
#      int edgeswap_count;
#      int t1_edgeswap_count;
#      int fix_count;
#      int unfix_count;
#      int notch_count;
 
#      /* flag words and bits */
#      int counts_reported;
#      int counts_changed;
     
#      #define equi_count_bit 0x00000001
#      #define weed_count_bit 0x00000002
#      #define edge_delete_count_bit 0x00000004
#      #define facet_delete_count_bit 0x00000008
#      #define edge_refine_count_bit 0x00000010
#      #define facet_refine_count_bit 0x00000020
#      #define notch_count_bit 0x00000040
#      #define vertex_dissolve_count_bit 0x00000080
#      #define edge_dissolve_count_bit 0x00000100
#      #define facet_dissolve_count_bit 0x00000200
#      #define body_dissolve_count_bit 0x00000400
#      #define vertex_pop_count_bit 0x00000800
#      #define edge_pop_count_bit 0x00001000
#      #define pop_tri_to_edge_count_bit 0x00004000
#      #define pop_edge_to_tri_count_bit 0x00008000
#      #define pop_quad_to_quad_count_bit 0x00010000
#      #define where_count_bit 0x00020000 
#      #define edgeswap_count_bit 0x00040000 
#      #define fix_count_bit 0x00080000 
#      #define unfix_count_bit 0x00100000 
#      #define t1_edgeswap_count_bit 0x00200000 
#      #define edge_reverse_count_bit 0x00400000
#      #define facet_reverse_count_bit 0x00800000
     

#      /* here follows stuff moved from independent globals to inside web
#          so as to be easily exported.  Previous global names are defined
#          to web fields elsewhere.
#          */
#      int gen_quant_count_w;
#      int gen_quant_alloc_w;
#      int gen_quant_free_head_w;
#      int global_count;
#      int maxglobals;  /* number allocated */
#      int perm_global_count;
#      int max_perm_globals;  /* number allocated */

#      int meth_inst_alloc_w;  /* number allocated */
#      int meth_inst_count_w;  /* number defined    */
#      int meth_inst_free_head_w;  /* start of free list */

#      /* global method instances, applying to every element of type */
#      int *global_meth_inst_w[NUMELEMENTS]; /* lists */
#      int global_meth_inst_count_w[NUMELEMENTS];
#      int global_meth_inst_alloc_w[NUMELEMENTS];

#      /* flags telling which quantity calculations necessary */
#      /* flag set for Q_ENERGY,Q_FIXED, or Q_INFO if any element
#          needs a quantity calculated */
#      int quant_flags_w[NUMELEMENTS];

#      DY_OFFSET dy_freestart_w;  /* initial block of freelist, 0 if none */
# #define dy_freestart web.dy_freestart_w
#      DY_OFFSET dy_globals_w;  /* global variable table */
#      struct global **dy_perm_globals_w;
#      DY_OFFSET dy_globalshash_w; /* hash list for global variables */

#      /* common */
#      int meth_attr[NUMELEMENTS] ; /* method instances list */
#      int mpi_export_attr[NUMELEMENTS] ; /* method instances list */
#   };

# extern struct webstruct web;
import torch
from constraint import Constraint,Volume
from energy import Energy,Area
from Geometric_Elements import Vertex,Edge,Face,Facet,Body
import numpy as np
import math
class webstruct:
    def __init__(self,vertex_list, edge_list, face_list,body_list=None,volume_constraint=None,energy:Energy = Area(),sdim = 3):#默认body constraint只有volume
        self.sdim = sdim #dimension of ambient space
        self.energy:Energy = energy
        self.VERTEXS:list[Vertex] = []
        self.EDGES:list[Edge]=[]
        self.FACES:list[Face]=[]
        self.FACETS:list[Facet]=[]
        self.BODIES:list[Body]=[]
        self.facet_diff = 1
        self.edge_diff = 1
        self.create_vertices(vertex_list)
        self.create_edges(edge_list)
        self.create_bodies(body_list,volume_constraint)  # Create bodies if provided
        self.create_facets(face_list)  # Create faces from the face list
        self.update_facet_of_body()  # Update facets of bodies after creation
        self.get_para()  # Print the number of vertices, edges, and facets after initialization
        
        if body_list is not None:
            for i in range(len(body_list)):
                self.get_body_para(i + 1)


    def update_facet_of_body(self):
        """Update the facets of each body based on the current FACETS."""
        for body in self.BODIES:
            body.update_facet_list(FACETS=self.FACETS)  # Update the list of facets in the body
            body.update_facet_sign(FACETS=self.FACETS)  # Update the signs of the facets in the body


    def create_vertices(self,vertex_list:list):
        """Create a list of Vertex objects from a list of coordinates and add it to VERTEXS."""
        # assert vertex_list[0].all()==0, "The first vertex must be at the origin (0, 0, 0)"
        for v in vertex_list:
            if type(v[0])==float:
                self.VERTEXS.append(Vertex(v[0],v[1],v[2]))
            else:
                is_fixed = v[1] if len(v) > 1 else False
                boundary = v[2] if len(v) > 2 else None
                self.VERTEXS.append(Vertex.from_vertex_list(v[0],is_fixed = is_fixed,boundary_func=boundary))



    def create_edges(self,edge_list:list[list[int]]):
        """Create a list of Edge objects from a list of vertex indices and add it to EDGES."""
        for e in edge_list:
            is_fixed = e[2] if len(e) > 2 else False
            boundary = e[3] if len(e) > 3 else None
            self.EDGES.append(Edge(vertex1=self.VERTEXS[e[0]-1], vertex2=self.VERTEXS[e[1]-1],is_fixed=is_fixed,boundary_func=boundary))
    
    def create_facets(self,face_list:list[list[int]]):
        for edge_list in face_list:
            vertex_list = []
            edges_list=[]
            ori=[]
            for edge in edge_list:
                if edge < 0:
                    e=self.EDGES[-edge-1]
                    edges_list.append(e)
                    ori.append(-1)
                    vertex_list.append(e.vertex2)
                else:
                    e=self.EDGES[edge-1]
                    edges_list.append(e)
                    ori.append(1)
                    vertex_list.append(self.EDGES[edge-1].vertex1)
            
            # Create face edges from the edge indices
            f = Face(vertexs=vertex_list,edges=edges_list,ori=ori)
            self.FACES.append(f)
            for body in self.BODIES:
                for fid in body.directed_face_list:
                    if abs(fid) == f.face_id:
                        body.add_facet_by_id(sign=1 if fid > 0 else -1)
                        break
            f.triangulation(self.FACETS,self.VERTEXS,self.EDGES)


    def create_bodies(self,body_list:list[list[int]],volume_constraint):
        """Create a list of Body objects from a list of facet indices."""
        for b in body_list:
            self.BODIES.append(Body(face_list=b))  # Initialize with empty faces
        for body,v_cons in zip(self.BODIES,volume_constraint):
            if v_cons is not None:
                body.add_constraints(Volume(float(v_cons)))

    def find_vertex_by_coordinates(self,x:float, y:float, z:float) -> Vertex:
        """Find a vertex by its coordinates."""
        for v in self.VERTEXS:
            if v.x == x and v.y == y and v.z == z:
                return v
        return None

    def find_edge_by_vertices(self,v1:Vertex, v2:Vertex) -> Edge:
        """Find an edge by its two vertices."""
        for e in self.EDGES:
            if (e.vertex1.vertex_id == v1.vertex_id and e.vertex2.vertex_id == v2.vertex_id) or (e.vertex1.vertex_id == v2.vertex_id and e.vertex2.vertex_id == v1.vertex_id):
                return e
        return None

    def update_vertex_coordinates(self,Verts:torch.Tensor):
        """Update the coordinates of all vertex."""
        # print(len(self.VERTEXS), len(Verts))
        # assert len(Verts) == len(VERTEXS), "The number of vertices must match the number of vertex coordinates provided."
        Verts=Verts.tolist()
        for i, vertex in enumerate(self.VERTEXS):
            x, y, z = Verts[i]
            vertex.move(x,y,z)


    def get_vertex_list(self) -> list[list[float]]:
        """Get the coordinates of all vertices."""
        return [[v.x, v.y, v.z] for v in self.VERTEXS]

    def get_vertex_tensor(self) ->torch.Tensor:
        """Get the coordinates of all vertices as a tensor."""
        if len(self.VERTEXS)==0:return None
        return torch.stack([v.coord for v in self.VERTEXS])

    def get_edge_list(self) -> list[list[int]]:
        """Get the list of edges as pairs of vertex IDs."""
        return [[e.vertex1.vertex_id, e.vertex2.vertex_id] for e in self.EDGES]

    def get_facet_list(self) -> list[list[int]]:
        """Get the list of facets as triplets of vertex IDs."""
        return [[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in self.FACETS]

    def get_facet_tensor(self) ->torch.Tensor:
        """Get the list of facets as a tensor of vertex coordinates."""
        return torch.tensor([[f.vertex1.vertex_id-1, f.vertex2.vertex_id-1, f.vertex3.vertex_id-1] for f in self.FACETS], dtype=torch.int64)

    def get_body_tensor(self)->torch.Tensor:
        """Get the list of bodies as a tensor of facet coordinates"""
        if len(self.BODIES)==0:return None
        return torch.stack([torch.tensor(np.array(b.get_facet_list())-self.facet_diff) for b in self.BODIES])

    def get_para(self):
        print(f"vertices:{len(self.VERTEXS)}, edges:{len(self.EDGES)}, facets:{len(self.FACETS)}")

    def get_body_para(self,bid):
        body:Body = self.BODIES[bid-1]
        print(f"body {bid}: volume:{body.compute_volume(self.FACETS)}, area:{body.get_surface_area(self.FACETS)}")

    def get_or_create_midpoint(self,v1:Vertex, v2:Vertex):
        x, y, z = (v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2
        mid = self.find_vertex_by_coordinates(x, y, z)
        if mid is None:
            e = self.find_edge_by_vertices(v1,v2)
            mid = Vertex(x, y, z,boundary_func=e.boundary_func)
            self.VERTEXS.append(mid)
        return mid

    def get_or_create_edge(self,v1, v2,boundary=None)->Edge:
        if self.find_edge_by_vertices(v1, v2) is None:
            self.EDGES.append(Edge(v1, v2,boundary_func=boundary))

    def single_facet_refinement(self,facet: Facet):
        v1, v2, v3 = facet.vertex1, facet.vertex2, facet.vertex3

        # Midpoints
        mid12 = self.get_or_create_midpoint(v1, v2)
        mid23 = self.get_or_create_midpoint(v2, v3)
        mid31 = self.get_or_create_midpoint(v3, v1)

        # Create edges (with duplication check)
        e1 = self.find_edge_by_vertices(v1,v2)
        assert e1 != None
        self.get_or_create_edge(v1, mid12,e1.boundary_func)
        self.get_or_create_edge(mid12,v2,e1.boundary_func)
        

        e2 = self.find_edge_by_vertices(v2,v3)
        assert e2 != None
        self.get_or_create_edge(v2, mid23,e2.boundary_func)
        self.get_or_create_edge(mid23,v3,e2.boundary_func)
        

        e3 = self.find_edge_by_vertices(v1,v3)
        assert e3 != None
        self.get_or_create_edge(v3, mid31,e3.boundary_func)
        self.get_or_create_edge(mid31,v1,e3.boundary_func)

        self.get_or_create_edge(mid12, mid23)
        self.get_or_create_edge(mid23, mid31)
        self.get_or_create_edge(mid31, mid12)

        # Create new facets
        self.FACETS.append(Facet(v1, mid12, mid31,facet._face_id))
        self.FACETS.append(Facet(mid12, v2, mid23,facet._face_id))
        self.FACETS.append(Facet(mid31, mid23, v3,facet._face_id))
        self.FACETS.append(Facet(mid12, mid23, mid31,facet._face_id))

    def refinement(self):
        n = len(self.FACETS)
        m = len(self.EDGES)

        # Copy of the original facets to avoid modifying while iterating
        original_facets = self.FACETS[:n]

        for facet in original_facets:
            self.single_facet_refinement(facet)

        del self.FACETS[:n]  # Remove original facets
        del self.EDGES[:m]   # Remove old edges
        self.facet_diff += n
        self.edge_diff += m

        self.update_facet_of_body()
        self.get_para()  # Print info

    def get_vertex_mask(self)->torch.Tensor:
        return  [[1-v.is_fixed, 1-v.is_fixed, 1-v.is_fixed] for v in self.VERTEXS]
    
    # def project_boundary_points_to_circle(self, Verts):
    #     """
    #     将所有在圆面边界上但非 fixed 的点，投影回边界圆上。
    #     假设边界圆函数为 (x, y, z) = (R * cos(θ), R * sin(θ), z)
    #     """
    #     with torch.no_grad():
    #         for idx, vertex in enumerate(self.VERTEXS):
    #             if vertex.is_fixed:
    #                 continue

    #             x, y, z = Verts[idx].tolist()
    #             theta = math.atan2(y, x)

    #             if vertex.boundary_func is not None:
    #                 Verts[idx]=torch.tensor(vertex.boundary_func.cal_cord([theta]))

    def b_proj(self,Verts):
        with torch.no_grad():
            for idx, vertex in enumerate(self.VERTEXS):
                if vertex.is_fixed:
                    continue

                if vertex.boundary_func is not None:
                    Verts[idx]=torch.tensor(vertex.boundary_func.b_proj(Verts[idx].data))

    # -u equiangulate
    def eartest(self, v1: Vertex, v2: Vertex, v3: Vertex, v4: Vertex)->bool:
        incident_edges: list[Edge] = []
        for edge in self.EDGES:
            if edge.vertex1 == v1 or edge.vertex2 == v1:
                incident_edges.append(edge)
        
        
        for edge in incident_edges:
            # check whether edge(v1, v2) already exists
            if edge.vertex1 != v2 and edge.vertex2 != v2:
                continue

            facet_vertex: list[Vertex] = []
            for facet in self.FACETS:
                v_ids = facet.vertex_idx
                if (v1.vertex_id - 1) in v_ids and (v2.vertex_id - 1) in v_ids:
                    facet_vertex.append({facet.vertex1, facet.vertex2, facet.vertex3, v1, v2})

            if {v3} in facet_vertex or {v4} in facet_vertex:
                return True

        return False
    
    def equiangulate_edge(self, edge: Edge) -> bool:
        if edge.is_fixed:
            return False
        
        vertex1 = edge.vertex1
        vertex2 = edge.vertex2
        vertex1_id = edge.vertex1.vertex_id
        vertex2_id = edge.vertex2.vertex_id

        matched_facet: list[Facet] = []
        facet_vertex = []
        for facet in self.FACETS:
            v_ids = facet.vertex_idx
            if (vertex1_id - 1) in v_ids and (vertex2_id - 1) in v_ids:
                matched_facet.append(facet)
                facet_vertex.append([facet.vertex1, facet.vertex2, facet.vertex3])
        
        if len(matched_facet) != 2:
            return False

        try:
            vertex3 = next(v for v in facet_vertex[0] if v.vertex_id != vertex1_id and v.vertex_id != vertex2_id)
            vertex4 = next(v for v in facet_vertex[1] if v.vertex_id != vertex1_id and v.vertex_id != vertex2_id)
        except StopIteration:
            # Degenerate triangle (edge belongs to triangle missing vertices)
            return False

        if vertex3.vertex_id == vertex4.vertex_id:
            return False
    
        shared = np.linalg.norm(vertex1.coord - vertex2.coord)
        facet1_edge1 = np.linalg.norm(vertex1.coord - vertex3.coord)
        facet1_edge2 = np.linalg.norm(vertex2.coord - vertex3.coord)
        facet2_edge1 = np.linalg.norm(vertex1.coord - vertex4.coord)
        facet2_edge2 = np.linalg.norm(vertex2.coord - vertex4.coord)

        cos_1 = (facet1_edge1**2 + facet1_edge2**2 - shared**2)/(facet1_edge1 * facet1_edge2)
        cos_2 = (facet2_edge1**2 + facet2_edge2**2 - shared**2)/(facet2_edge1 * facet2_edge2)
        if (cos_1 + cos_2) > -0.001:
            return False
        
        if self.eartest(vertex3, vertex4, vertex1, vertex2):
            return False

        # do swap
        # facet 1: v1 v2 v3->v1 v3 v4
        matched_facet[0].vertex1 = vertex1
        matched_facet[0].vertex2 = vertex3
        matched_facet[0].vertex3 = vertex4
        matched_facet[0].vertex_idx = [vertex1.vertex_id-1, vertex3.vertex_id-1, vertex4.vertex_id-1]
        matched_facet[0].volume = matched_facet[0].compute_volume()

        # facet 2: v1 v2 v4->v2 v3 v4
        matched_facet[1].vertex1 = vertex2
        matched_facet[1].vertex2 = vertex3
        matched_facet[1].vertex3 = vertex4
        matched_facet[1].vertex_idx = [vertex2.vertex_id-1, vertex3.vertex_id-1, vertex4.vertex_id-1]
        matched_facet[1].volume = matched_facet[1].compute_volume()

        # delete edge
        self.EDGES.remove(edge)
        # add edge
        if self.find_edge_by_vertices(vertex3, vertex4) == None:
            self.EDGES.append(Edge(vertex3, vertex4))

        return True
    
    def equiangulate(self):
        count = 0
        edges_copy: list[Edge] = []
        for edge in self.EDGES:
            edges_copy.append(edge)
        for edge in edges_copy:
            count += self.equiangulate_edge(edge)
        return count
    
    # -t remove tiny edges
    def delete_short_edges(self, min_edge_length=1e-5):
        to_remove = []
        for edge in self.EDGES:
            if edge.length() < min_edge_length:
                v1, v2 = edge.vertex1, edge.vertex2
                if v1.is_fixed or v2.is_fixed:
                    continue  # Skip fixed vertices
                # 合并点: 使用较旧的v1保留，删除v2
                new_x = (v1.x + v2.x) / 2
                new_y = (v1.y + v2.y) / 2
                new_z = (v1.z + v2.z) / 2
                v1.move(new_x, new_y, new_z)
                # 替换所有面中v2为v1
                for facet in self.FACETS:
                    if facet.vertex1 == v2:
                        facet.vertex1 = v1
                    if facet.vertex2 == v2:
                        facet.vertex2 = v1
                    if facet.vertex3 == v2:
                        facet.vertex3 = v1
                    facet.vertex_idx = [v.vertex_id - 1 for v in [facet.vertex1, facet.vertex2, facet.vertex3]]
                # 移除相关边
                to_remove.append(edge)
                for other in self.EDGES:
                    if other != edge and (other.vertex1 == v2 or other.vertex2 == v2):
                        to_remove.append(other)
                # 删除该点
                if v2 in self.VERTEXS:
                    self.VERTEXS.remove(v2)

        # 实际移除边（去重）
        for edge in set(to_remove):
            if edge in self.EDGES:
                self.EDGES.remove(edge)
        
        # 删除非法面（重复点）
        self.FACETS = [f for f in self.FACETS if len(set([f.vertex1.vertex_id, f.vertex2.vertex_id, f.vertex3.vertex_id])) == 3]