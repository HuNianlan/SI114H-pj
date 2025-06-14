import torch
class Boundary:
    '''A class to define the boundary function on vertex or edge'''
    def __init__(self,num_par,func = None):
        self.num_par = num_par
        self.func = func
        self.estimated_par = None
    def cal_cord(self,par:list):
        assert len(par) ==self.num_par
        return self.func(par)
    def b_proj(cord):
        pass

from math import pi,cos,sin,hypot
class Ring_XY(Boundary):
    def __init__(self, center,radius,z_cord=None):
        self.center = center
        self.radius = radius
        self.z_cord = z_cord
        def generate_fun(rho):
            x0= center[0]
            y0= center[1]
            x = radius *cos(rho[0])+x0
            y = radius *sin(rho[0])+y0
            z = z_cord
            return [x,y,z]
        super().__init__(1, func = generate_fun)
    
    def b_proj(self,cord):
        x, y, z = cord
        x0, y0 = self.center

        # 向量从圆心指向该点
        dx = x - x0
        dy = y - y0
        dist = hypot(dx, dy)  # sqrt(dx^2 + dy^2)

        if dist == 0:
            # 如果点正好在圆心上，方向不定，默认放在 (x0 + r, y0)
            proj_x = x0 + self.radius
            proj_y = y0
        else:
            proj_x = x0 + self.radius * dx / dist
            proj_y = y0 + self.radius * dy / dist
        if self.z_cord==None:
            proj_z = z
        else:
            proj_z=self.z_cord

        return (proj_x, proj_y, proj_z)

        
    

class Ring_XZ(Boundary):
    def __init__(self, center,radius,y_cord=None):
        self.center = center
        self.radius = radius
        self.y_cord = y_cord
        def generate_fun(rho):
            x0= center[0]
            y0= center[1]
            x = radius *cos(rho[0])+x0
            z = radius *sin(rho[0])+y0
            y = y_cord
            return [x,y,z]
        super().__init__(1, func = generate_fun)
    
    def b_proj(self,cord):
        x, y, z = cord
        x0, z0 = self.center

        # 向量从圆心指向该点
        dx = x - x0
        dz = z - z0
        dist = hypot(dx, dz)  # sqrt(dx^2 + dy^2)

        if dist == 0:
            # 如果点正好在圆心上，方向不定，默认放在 (x0 + r, y0)
            proj_x = x0 + self.radius
            proj_z = z0
        else:
            proj_x = x0 + self.radius * dx / dist
            proj_z = z0 + self.radius * dz / dist
        if self.y_cord ==None:
            proj_y = y
        else:
            proj_y = self.y_cord

        return (proj_x, proj_y,proj_z)

class LevelSetConstraint_Plane(Boundary):
    def __init__(self, normal_vector, point_on_plane):
        """
        :param normal_vector: 平面的法向量，如 [0,0,1] 表示 xy-平面
        :param point_on_plane: 平面上的一个点，如 [0,0,0]
        """
        self.normal = normal_vector  # 法向量 [a,b,c]
        self.point = point_on_plane  # 平面上点 [x0,y0,z0]
        # 平面方程: a(x-x0) + b(y-y0) + c(z-z0) = 0

        # 无参数化函数，直接继承 Boundary
        super().__init__(0, func=None)

    def b_proj(self, cord):
        """
        将点 cord = [x,y,z] 投影到平面上
        """
        x, y, z = cord
        a, b, c = self.normal
        x0, y0, z0 = self.point

        # 计算点到平面的距离
        distance = a*(x - x0) + b*(y - y0) + c*(z - z0)
        denom = a**2 + b**2 + c**2

        # 投影点坐标
        proj_x = x - a * distance / denom
        proj_y = y - b * distance / denom
        proj_z = z - c * distance / denom

        return (proj_x, proj_y, proj_z)