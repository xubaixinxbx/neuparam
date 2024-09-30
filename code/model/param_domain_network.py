import torch.nn as nn
import torch
from model.embedder import *

from model.network import ImplicitNetwork

class ParamDomainNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.param_domain_conf = conf.get_config('param_domain_network') 
        self.plot_template = False
        self.device = 'gpu'
        self.implicit_net = ImplicitNetwork(temp_feature_dim=0,\
                                sdf_bounding_sphere=0,\
                                **self.param_domain_conf)
        
        self.implicit_net.update_embed_fn(1000) # set a default value, frequency of position encoding will not change along with training
            
    def forward(self, input, embeddings=None):
        # Parse model input
        surface_points = input['points'].squeeze(0)

        points_all = surface_points
        points_all.requires_grad_(True)
        points_temp = points_all
        
        self.device = points_all.device
        sdf_all = self.implicit_net(points_temp)
        gradients = self.get_gradient(x=points_all, y=sdf_all)
        
        output = {
            'sdf': sdf_all,
        } 
        if self.training:
            output['grad_theta'] = gradients
        return output
    
    
    def get_gradient(self, x, y):
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_uv_coords(self, x):
        # x:  3d-dims space points
        uv_coords = torch.zeros((x.shape[0], 2)).to(x.device)
        # use steoro projection
        # uv_coords[:, 0] = x[:, 0] / (1 - x[:, 2])
        # uv_coords[:, 1] = x[:, 1] / (1 - x[:, 2])
        # use polar-coords
        uv_coords[:, 0] = torch.arctan(torch.sqrt(x[:,0]**2 + x[:,1]**2) / x[:,2])
        uv_coords[:, 1] = torch.arctan(x[:,1] / x[:,0])
        uv_coords = uv_coords / torch.pi + 0.5 #range [0, 1]
        return uv_coords

    def set_plot_template(self, plot_template):
        self.plot_template = plot_template

    def get_sdf_vals(self, x, shape_code=None):
        sdf = self.implicit_net(x)
        return sdf
    
    def get_embedding(self, shape_idx=[-1],color_idx=[-1]):
        color_code = None
        shape_code = None
        embeddings = {
            'shape_code' : shape_code,
            'color_code' : color_code,
        }
        return embeddings