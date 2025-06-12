# import torch
class Boundary:
    '''A class to define the boundary function on vertex or edge'''
    def __init__(self,num_par,func = None):
        self.num_par = num_par
        self.func = func
    def cal_cord(self,par:list):
        assert len(par) ==self.num_par
        return self.func(par)
    
