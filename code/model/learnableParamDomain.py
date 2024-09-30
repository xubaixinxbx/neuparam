import torch.nn as nn
import numpy as np
import torch
from model.embedder import *

class LearnableSphere(nn.Module):
    def __init__(self, centroid=[0.1,0.1,0.1], scale=1):
        super().__init__()
        self.register_parameter('centroid', nn.Parameter(torch.tensor(centroid,requires_grad=True,dtype=torch.float32)))
        self.register_parameter('scale', nn.Parameter(torch.tensor(scale,requires_grad=True,dtype=torch.float32)))
        
    def forward(self, p):
        return torch.norm(p-self.centroid,dim=-1, keepdim=True) - self.scale
    
    def get_centroid(self):
        return self.centroid
    
    def get_scale(self):
        return self.scale
    
    def set_centroid(self, centroid):
        self.centroid.data.fill_(centroid)
    
    def set_scale(self, scale):
        self.scale.data.fill_(scale)

class LearnableCube(nn.Module):
    def __init__(self, centroid=[0.1,0.1,0.1], scale=[0.1,0.1,0.1], pow=6):
        super().__init__()
        # centroid: [x,y,z], scale: [sx, sy, sz]
        self.register_parameter('centroid', nn.Parameter(torch.tensor(centroid,requires_grad=True,dtype=torch.float32)))
        self.register_parameter('scale', nn.Parameter(torch.tensor(scale,requires_grad=True,dtype=torch.float32)))
        self.pow = pow
        self.eps = 1.0e-6
        self.lmd =  100
        
    def forward(self, p):
        # p[n,3]
        res, _ = torch.max(torch.cat([torch.abs(p[:,0:1] - self.centroid[0]) - self.scale[0], \
            torch.abs(p[:,1:2] - self.centroid[1]) - self.scale[1], \
                torch.abs(p[:,2:3] - self.centroid[2]) - self.scale[2]],dim=-1), dim=-1, keepdim=True)
        return res
    
    def get_centroid(self):
        return self.centroid
    
    def get_scale(self):
        return self.scale
    
    def set_centroid(self, centroid):
        self.centroid.data.fill_(centroid)
    
    def set_scale(self, scale):
        self.scale.data.fill_(scale)

class LearnableParamDomain(nn.Module):
    def __init__(self, num_cubes=8):
        super().__init__()
        self.num_cubes = num_cubes
        self.lmd = 100
        # setattr(self, f'cube1', LearnableCube(centroid=[-0.0039, -0.4202, -0.2189], scale=[0.1965, 0.6929, 0.1403]))
        # setattr(self, f'cube0', LearnableCube(centroid=[0.0678, -0.9548, -0.2604], scale=[0.7791, 0.3895, 0.2990]))
        # setattr(self, f'cube2', LearnableCube(centroid=[-0.0094, 0.1197, -0.0571], scale=[0.2597, 0.4348, 0.3492]))

        # truck44
        # setattr(self, f'cube0', LearnableCube(centroid=[0.46232694387435913, -0.02399148792028427, 0.04755323752760887], scale=[0.41128838062286377, 0.31836360692977905, 0.3889177143573761]))
        # setattr(self, f'cube1', LearnableCube(centroid=[-0.12232348322868347, -0.02532101981341839, -0.030581874772906303], scale=[0.7685419917106628, 0.33259618282318115, 0.32174840569496155]))
        # truck29
        # setattr(self, f'cube0', LearnableCube(centroid=[0.15731440484523773, 0.06828219443559647, -0.05727982521057129], scale=[0.7921415567398071, 0.43673884868621826, 0.3009502589702606]))
        # setattr(self, f'cube1', LearnableCube(centroid=[-0.21492761373519897, 0.06275720149278641, 0.108684241771698], scale=[0.6373299360275269, 0.37178245186805725, 0.4129398763179779]))
        # ornament=3
        # setattr(self, f'cube0', LearnableCube(centroid=[-0.02808156982064247, 0.04969673976302147, -0.06293125450611115], scale=[0.6542162299156189, 0.6491073966026306, 0.8308215141296387]))
        setattr(self, f'cube0', LearnableSphere(centroid=[0, 0.0, 0.0], scale=1.0))
        # ornament=4
        # setattr(self, f'cube0', LearnableCube(centroid=[-0.028391456231474876, -0.04946647956967354, -0.0790281742811203], scale=[0.8824332356452942, 0.8929451704025269, 0.8265908360481262]))
    
    def forward(self, p):
        #[n,3]
        res = 0
        cube_res = []
        for i in range(0, self.num_cubes):
            cubes = getattr(self, f'cube{i}')
            res = -self.lmd*cubes(p)
            cube_res.append(res)
        cube_res = torch.cat(cube_res, dim=-1) #[n,num_cubes]
        res = -1/self.lmd * torch.logsumexp(cube_res, dim=-1, keepdim=True) #[n,1]
        return res
    
    def max_pc(self, p):
        cube_res = []
        for i in range(0, self.num_cubes):
            cubes = getattr(self, f'cube{i}')
            res = cubes(p)
            cube_res.append(res)
        cube_res = torch.cat(cube_res, dim=-1) #[n,num_cubes]
        res, _ = torch.max(-cube_res,dim=-1,keepdim=True)
        return -res