# import torch
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

        
    

