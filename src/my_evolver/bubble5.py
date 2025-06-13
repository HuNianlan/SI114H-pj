def parse_data(input_data):
    lines = input_data.strip().split('\n')
    # 初始化各个部分
    vertices = []
    edges = []
    faces = []
    bodies = []
    volumes = []
    
    # 标志位控制
    vertex_section = False
    edge_section = False
    face_section = False
    body_section = False
    
    for line in lines:
        line = line.split('//')[0].strip()
        if not line.strip():
            continue
        if line.startswith("vertices"):
            vertex_section = True
            edge_section = False
            face_section = False
            body_section = False
            continue
        elif line.startswith("edges"):
            vertex_section = False
            edge_section = True
            face_section = False
            body_section = False
            continue
        elif line.startswith("faces"):
            vertex_section = False
            edge_section = False
            face_section = True
            body_section = False
            continue
        elif line.startswith("bodies"):
            vertex_section = False
            edge_section = False
            face_section = False
            body_section = True
            continue
        elif vertex_section:
            # 解析顶点数据，去除编号
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            # print(parts[3])
        elif edge_section:
            # 解析边数据，去除编号
            parts = line.split()
            edges.append([int(parts[1]), int(parts[2])])
        elif face_section:
            # 解析面数据，去除编号
            parts = line.split()
            faces.append([int(parts[1]), *[int(x) for x in parts[2:]]])
        elif body_section:
            # 解析体数据，提取volume到单独的list
            parts = line.split()
            bodies.append([*map(int, parts[1:-2])])  # 去掉编号和最后一列volume
            volumes.append(float(parts[-1]))  # 仅保存volume
    
    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "bodies": bodies,
        "volumes": volumes
    }

# 示例输入
input_data = """
vertices //     coordinates
  1      0.877383  0.000000  1.000000
  2      0.877383  0.000000  0.000000
  3      0.877383  0.000000  -1.000000
  4      2.057645  0.000000  -1.000000
  5      2.057645  0.000000  1.000000
  6      -0.438691  -0.759836  1.000000
  7      -0.438691  -0.759836  0.000000
  8      -0.438691  -0.759836  -1.000000
  9      -1.028822  -1.781973  -1.000000
 10      -1.028822  -1.781973  1.000000
 11      -0.438691  0.759836  1.000000
 12      -0.438691  0.759836  0.000000
 13      -0.438691  0.759836  -1.000000
 14      -1.028822  1.781973  -1.000000
 15      -1.028822  1.781973  1.000000

edges  // endpoints
  1       1    2
  2       2    3
  3       3    4
  4       4    5
  5       5    1
  6       1    6
  7       2    7
  8       3    8
  9       4    9
 10       5   10
 11       6    7
 12       7    8
 13       8    9
 14       9   10
 15       10    6
 16       6   11
 17       7   12
 18       8   13
 19       9   14
 20      10   15
 21      11   12
 22      12   13
 23      13   14
 24      14   15
 25      15   11
 26      11    1
 27      12    2
 28      13    3
 29      14    4
 30      15    5

faces //   edges    
  1      1   2   3   4   5 
  2      6  11  -7  -1 
  3      7  12  -8  -2 
  4      8  13  -9  -3 
  5     10 -14  -9   4 
  6      6 -15 -10   5 
  7     11  12  13  14  15 
  8     16  21 -17 -11 
  9     17  22 -18 -12 
 10     18  23 -19 -13 
 11     20 -24 -19  14 
 12     16 -25 -20  15 
 13     21  22  23  24  25 
 14     26   1 -27 -21 
 15     27   2 -28 -22 
 16     28   3 -29 -23 
 17     30  -4 -29  24 
 18     26  -5 -30  25 
 19    -26  -16   -6  
 20    -27  -17   -7  
 21    -28  -18   -8  

bodies    //     facets 
  1     -1   -2   -3   -4    5    6    7     volume  3.000000  
  2     -7   -8   -9  -10   11   12   13     volume  3.000000  
  3    -13  -14  -15  -16   17   18    1     volume  3.000000  
  4     19    2    8   14  -20     volume  1.000000
  5     20    3    9   15  -21     volume  1.000000
"""

# 调用解析函数
data = parse_data(input_data)

vertices = data['vertices']
edges = data['edges']
faces = data['faces']
bodies = data['bodies']
volume_constraint = data['volumes']

from iterate import iterate
########################################################################################################
from web import webstruct
web = webstruct(vertices, edges, faces, bodies,volume_constraint)
for i in range(3):
    iterate(web, num_iterations=500)
    web.refinement()
    web.delete_short_edges(0.16)

########################################################################################################
from visualization import plot_mesh

plot_mesh(web.get_vertex_list(), web.get_facet_list(), "Optimized Mesh")