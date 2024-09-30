import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler, UniformSampler

class DeformNetwork(nn.Module):
    def __init__(self, shape_code_dim, dims, d_in=3, d_out=3, \
                multires=0, weight_norm=True, deform_feature_dim=128,
                ):
        super().__init__()
        d_out = d_out + deform_feature_dim
        dims = [d_in + shape_code_dim] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + shape_code_dim

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, shape_latent_code):
        if self.embed_fn is not None:
            x = self.embed_fn(x)
        if x.shape[0] != shape_latent_code.shape[0] and shape_latent_code.shape[0] == 1:
            shape_latent_code = shape_latent_code.repeat(x.shape[0],1)
        assert shape_latent_code.shape[0] == x.shape[0], print('in deform net, shape_code.dim != x.dim')
        x =  torch.cat([x, shape_latent_code], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        return x
    
    def update_embed_fn(self, alpha):
        if self.embed_fn is not None:
            self.embed_fn.update_alpha(alpha)

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            temp_feature_dim,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            input_feat_dim=0,
    ):
        super().__init__()
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in+input_feat_dim] + dims + [d_out + temp_feature_dim]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - d_in
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, shape_latent_code=None):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        if shape_latent_code is not None:
            if input.shape[0] != shape_latent_code.shape[0] and shape_latent_code.shape[0] == 1:
                shape_latent_code = shape_latent_code.repeat(input.shape[0],1)
            input = torch.cat([input, shape_latent_code], 1)
        
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def update_embed_fn(self, alpha):
        if self.embed_fn:
            self.embed_fn.update_alpha(alpha)

class DeformTempGeoNetwork(nn.Module):
    def __init__(self,conf,**kwargs):
        super().__init__()
        self.deform_conf = conf.get_config('deform_network')
        self.implicit_conf = conf.get_config('implicit_network')
        self.sphere_scale = self.implicit_conf.get_float('sphere_scale')
        self.temp_feature_dim = kwargs['temp_feature_dim']
        shape_code_dim = kwargs['shape_code_dim']
        self.white_bkgd = kwargs['white_bkgd']
        self.deform_feature_dim = self.deform_conf.get_int('deform_feature_dim',default=0)
        self.scene_bounding_sphere = kwargs['scene_bounding_sphere']
        self.sdf_bounding_sphere = 0.0 if self.white_bkgd else self.scene_bounding_sphere
        self.plot_template = False

        self.deform_net = DeformNetwork(shape_code_dim=shape_code_dim,\
                                        **self.deform_conf)

        self.implicit_net = ImplicitNetwork(temp_feature_dim=self.temp_feature_dim,\
                                sdf_bounding_sphere=self.sdf_bounding_sphere,\
                                **self.implicit_conf)
    
    def set_plot_template(self, plot_template=False):
        self.plot_template = plot_template
    
    def update_embed_fn(self, alpha):
        self.deform_net.update_embed_fn(alpha)
        self.implicit_net.update_embed_fn(alpha)

    def get_deform_grad(self, x, delta_x):
        u = delta_x[:, 0]
        v = delta_x[:, 1]
        w = delta_x[:, 2]
        d_output = torch.ones_like(u, requires_grad=False, device=u.device)
        grad_u = torch.autograd.grad(u, x, grad_outputs=d_output, retain_graph=True, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, x, grad_outputs=d_output, retain_graph=True, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, x, grad_outputs=d_output, retain_graph=True, create_graph=True)[0]
        grad_deform = torch.stack([grad_u, grad_v, grad_w], dim=-1)
        return grad_deform

    def forward(self, x, shape_code):
        delta_x_with_feature = self.deform_net(x, shape_code)
        delta_x = delta_x_with_feature[:,:3]
        deform_feature = delta_x_with_feature[:,3:]
            
        if self.plot_template:
            ref_points = x
        else:
            ref_points = x + delta_x

        sdf_with_feature = self.implicit_net(ref_points)
        template_feature = sdf_with_feature[:,1:]

        geometry_feature = torch.cat([deform_feature, template_feature], -1)
        sdf_with_feature = torch.cat([sdf_with_feature[:,:1], geometry_feature], -1) #new feature fed into render-net

        if self.plot_template:
            # for rendering rgb, if template, we dumplicate Template feature only when they are in the same dim
            # sdf_with_feature = torch.cat([sdf_with_feature[:,:1], template_feature, template_feature], -1)
            pass
        
        return sdf_with_feature, delta_x
    
    def gradient(self, x, shape_code):
        x.requires_grad_(True)
        y, delta_x = self.forward(x, shape_code)
        y = y[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients
    
    def get_outputs(self, x, shape_code):
        if shape_code.shape[0] == 1:
            shape_code = shape_code.repeat(x.shape[0],1)
        x.requires_grad_(True)
        output, delta_x = self.forward(x, shape_code)
        sdf = output[:,:1]

        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        
        grad_deform = None
        if self.training:
            # grad_deform = self.get_deform_grad(x, delta_x=delta_x_s[:,:3])
            deform_output = torch.ones_like(delta_x, requires_grad=False, device=delta_x.device)
            grad_deform = torch.autograd.grad(delta_x, x, grad_outputs=deform_output, retain_graph=True, create_graph=True)[0]

        sdf_gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, [delta_x, grad_deform], feature_vectors, sdf_gradients
    
    def get_sdf_vals(self, x, shape_code):
        if shape_code.shape[0] == 1:
            shape_code = shape_code.repeat(x.shape[0],1)

        sdf_with_feature, delta_x = self.forward(x, shape_code)
        sdf = sdf_with_feature[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires=0,
            skip_in=(),
            acti_flag=True
    ):
        super().__init__()
        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.skip_in = skip_in
        self.embedview_fn = None
        self.embed_fn = None
        self.acti_flag = acti_flag

        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            if self.mode == 'idr':
                dims[0] += (input_ch -3)
            else:
                dims[0] += (input_ch - d_in)
        if self.mode == 'idr' and multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch -3)

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                in_dim = dims[l] + dims[0]
            else:
                in_dim = dims[l]
            lin = nn.Linear(in_dim, dims[l + 1])
            # if l + 1 in self.skip_in:
            #     out_dim = dims[l + 1] - dims[0]
            # else:
            #     out_dim = dims[l + 1]
            # lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals=None, view_dirs=None, feature_vectors=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        
        rendering_input = points
        if self.mode == 'idr':
            if normals is not None:
                rendering_input = torch.cat([rendering_input, normals], dim=-1)
            if view_dirs is not None:
                rendering_input = torch.cat([rendering_input, view_dirs], dim=-1)
            if feature_vectors is not None:
                rendering_input = torch.cat([rendering_input, feature_vectors], dim=-1)

        elif self.mode == 'nerf':
            if feature_vectors is not None:
                rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
            else:
                rendering_input = view_dirs
        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l + 1 in self.skip_in:
                x = torch.cat([x, rendering_input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        if self.acti_flag:
            x = self.sigmoid(x)
        return x

    def update_embed_fn(self, alpha):
        if self.embed_fn is not None:
            self.embed_fn.update_alpha(alpha)
        if self.embedview_fn is not None:
            self.embedview_fn.update_alpha(alpha)
