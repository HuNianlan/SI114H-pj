# visualization.py
import platform
import numpy as np

class MeshVisualizer:
    def __init__(self):
        self.system = platform.system()
        self.plotter = None
        
    def visualize(self, vertices, faces, title="Mesh"):
        """自动选择适合当前系统的可视化方式"""
        try:
            if self.system == "Linux":
                self._visualize_linux(vertices, faces, title)
            else:
                self._visualize_desktop(vertices, faces, title)
        except Exception as e:
            print(f"可视化失败，使用备用方案: {str(e)}")
            self._visualize_fallback(vertices, faces, title)

    def _visualize_desktop(self, vertices, faces, title):
        """桌面环境使用Polyscope"""
        import polyscope as ps
        ps.init()
        ps.set_ground_plane_mode('none')
        ps.register_surface_mesh(title, np.array(vertices), np.array(faces))
        ps.show()

    def _visualize_linux(self, vertices, faces, title):
        """Linux环境使用Matplotlib"""
        self._visualize_with_matplotlib(vertices, faces, title)

    def _visualize_with_matplotlib(self, vertices, faces, title):

        # ============= 替换为Matplotlib可视化 =============
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from utils import get_facet_list1,get_vertex_list1,get_facet_list,get_vertex_list

        vertices = np.array(get_vertex_list())
        faces = np.array(get_facet_list())

        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 准备面数据
        mesh_faces = []
        for face in faces:
            if len(face) == 3:  # 三角形
                mesh_faces.append(vertices[face])
            elif len(face) == 4:  # 四边形分割为两个三角形
                mesh_faces.append(vertices[face[[0,1,2]]])
                mesh_faces.append(vertices[face[[0,2,3]]])

        # 添加网格面片
        mesh = Poly3DCollection(
            mesh_faces,
            alpha=0.8,
            linewidths=0.5,
            edgecolor='k',
            facecolor='lightblue'
        )
        ax.add_collection3d(mesh)

        # 设置坐标轴
        min_coord, max_coord = vertices.min(), vertices.max()
        ax.set_xlim(min_coord, max_coord)
        ax.set_ylim(min_coord, max_coord)
        ax.set_zlim(min_coord, max_coord)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()
        # """纯Python后备方案"""
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        
        # # 准备面数据
        # mesh_faces = []
        # for face in faces:
        #     if len(face) == 3:
        #         mesh_faces.append(vertices[face])
        #     elif len(face) == 4:
        #         mesh_faces.append(vertices[face[[0,1,2]]])
        #         mesh_faces.append(vertices[face[[0,2,3]]])
        
        # mesh = Poly3DCollection(
        #     mesh_faces,
        #     alpha=0.8,
        #     linewidths=0.5,
        #     edgecolor='k',
        #     facecolor='lightblue'
        # )
        # ax.add_collection3d(mesh)
        # ax.set_title(title)
        # plt.show()

    def _visualize_fallback(self, vertices, faces, title):
        """终极后备方案：保存文件"""
        print(f"无法显示图形，已将网格保存为 {title}.obj")
        with open(f"{title}.obj", 'w') as f:
            for v in vertices:
                f.write(f"v {' '.join(map(str, v))}\n")
            for face in faces:
                f.write(f"f {' '.join(str(i+1) for i in face)}\n")

# 创建全局可视化实例
visualizer = MeshVisualizer()

def plot_mesh(vertices, faces, title="Mesh"):
    """对外接口函数"""
    visualizer.visualize(vertices, faces, title)