
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
