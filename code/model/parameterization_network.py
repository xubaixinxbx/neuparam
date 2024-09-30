import torch.nn as nn
import numpy as np
import torch
from model.embedder import *
import mcubes
import trimesh
from utils import rend_util
from model.network import ImplicitNetwork, RenderingNetwork, LaplaceDensity, ErrorBoundSampler
from utils.color_corres import load_cube_from_single_texture, sample_cubemap
from model.learnableParamDomain import LearnableParamDomain

class ParamNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.dim_id_shape = conf.get_int('dim_identity_shape', default=0)
        self.dim_id_color = conf.get_int('dim_identity_color', default=0)
        self.param_conf = conf.get_config('param_network') 
        self.deform_conf = self.param_conf.get_config('deform_network')
        self.implicit_conf = self.param_conf.get_config('implicit_network')
        self.tex_conf = conf.get_config('texture_network', default=None)
        self.shade_conf = conf.get_config('shading_network', default=None)
        self.density_conf = conf.get_config('density',default=None)
        self.ray_sample_conf = conf.get_config('ray_sampler',default=None)

        self.plot_template = False
        self.device = 'gpu'

        self.num_id = conf.get_int('num_identity', default=1)
        self.laplace_flag = conf.get_bool('laplace_flag', default=False)
        self.learnable_domain_flag = conf.get_bool('learnable_domain', default=False)

        # define params to extract from marching cube
        self.grid_res_for_lap = conf.get_int('grid_res', default=64)
        bound_min = conf.get_float('bound_min', default=-1.0)
        bound_max = conf.get_float('bound_max', default=1.0)
        self.bound_min = [bound_min, bound_min, bound_min]
        self.bound_max = [bound_max, bound_max, bound_max]
        
        self.use_inverse_flag = conf.get_bool('use_cycle', default=True)
        self.ref_to_tex_path = conf.get_string('ref_to_tex_path', default=None)
        
        if self.ref_to_tex_path is not None:
            tex_images = load_cube_from_single_texture(self.ref_to_tex_path)
            self.tex_images_ref = torch.tensor(tex_images).float().cuda()[...,:3]


        self.deform_net = ImplicitNetwork(input_feat_dim=self.dim_id_shape,\
                            sdf_bounding_sphere=0,\
                            **self.deform_conf)
        if self.use_inverse_flag:
            self.deform_inv_net = ImplicitNetwork(input_feat_dim=self.dim_id_shape,\
                                sdf_bounding_sphere=0,\
                                **self.deform_conf)

        self.density = LaplaceDensity(**self.density_conf)
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **self.ray_sample_conf)
        self.id_color_embedding = nn.Embedding(num_embeddings=self.num_id, embedding_dim=self.dim_id_shape)
        nn.init.normal_(self.id_color_embedding.weight, mean=0, std=0.01)
        self.tex_network = RenderingNetwork(feature_vector_size=self.dim_id_color,\
                        **self.tex_conf)
        self.shade_network = RenderingNetwork(feature_vector_size=self.dim_id_color,\
                        **self.shade_conf)
        pe_alpha = conf.get_float('pe_alpha', default=1000.0)
        self.update_embed_fn(pe_alpha)


        self.id_shape_embedding = nn.Embedding(num_embeddings=self.num_id, embedding_dim=self.dim_id_shape)
        nn.init.normal_(self.id_shape_embedding.weight, mean=0, std=0.01)
        if not self.learnable_domain_flag:
            self.implicit_net = ImplicitNetwork(temp_feature_dim=0,\
                                    sdf_bounding_sphere=0,\
                                    **self.implicit_conf)
            if self.implicit_net.embed_fn is not None:
                self.implicit_net.update_embed_fn(pe_alpha)
        else:
            self.num_cubes = conf.get_int('num_cubes', default=0)
            assert self.num_cubes > 0, print("num of cubes is defined < 1")

            self.implicit_net = LearnableParamDomain(num_cubes=self.num_cubes)
        

    def forward(self, input, embeddings=None):
        # Parse model input

        id = input["id"]
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        self.device = pose.device

        if embeddings is None:
            if 'color_id' in input:
                color_id = input['color_id']
            else:
                color_id = id
            embeddings = self.get_embedding(shape_idx=id, color_idx=color_id)
        shape_latent_code = embeddings['shape_code']
        color_latent_code = embeddings['color_code']
        
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik, z_samples = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, shape_latent_code)

        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)
        self.N_samples = N_samples


        # neural parameterization process
        points_all = points_flat
        points_all.requires_grad_(True)
        d2t_offset = self.deform_net(points_all,shape_latent_code=shape_latent_code)
        points_temp = points_all + d2t_offset

        sdf_all = self.implicit_net(points_temp)
        gradients = self.get_gradient(x=points_all, y=sdf_all)

        # inverse neural parameterization process
        if self.use_inverse_flag:
            t2d_offset = self.deform_inv_net(points_temp, shape_latent_code=shape_latent_code)
            points_cir = points_temp + t2d_offset
    
        # rendering process
        weights = self.volume_rendering(z_vals, sdf_all)
        color_latent_code_repeat = color_latent_code.repeat(points_flat.shape[0],1)
        # refer to nep for dynamic editing
        if self.ref_to_tex_path is not None:
            tex_flat = self.get_vcolor_from_texmap(points_temp)
            appearance_view_flat = self.shade_network(points=points_temp, view_dirs=dirs_flat, normals=gradients, feature_vectors=color_latent_code_repeat)
        else:
            tex_flat = self.tex_network(points=points_temp, normals=gradients, feature_vectors=color_latent_code_repeat)
            appearance_view_flat = self.shade_network(points=points_temp, view_dirs=dirs_flat, normals=gradients, feature_vectors=color_latent_code_repeat)
        
        rgb_flat = (tex_flat * torch.exp(appearance_view_flat)).clamp(0,1)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        # prepare output
        output = {
            'rgb_values': rgb_values,
        }
        if self.shade_network is not None:
            output['shading_values'] = appearance_view_flat
            
        output['deform'] = d2t_offset
        if self.use_inverse_flag:
            output['deform_inv'] = t2d_offset
            output['points_cir'] = (points_cir-points_all) * weights.reshape(-1, 1)

        if self.training:
            output['grad_theta'] = gradients

            if self.learnable_domain_flag:
                domain_sdf = self.implicit_net(points_flat)
                output['domain_diff'] = domain_sdf - sdf_all
                
            output['code'] = {
                'shape_code': shape_latent_code, 
                'color_code': color_latent_code
            }
            if self.laplace_flag:
                # the process of sampling points doesnot require differentiable
                points_surface, neighbors_indices, lap_weights = self.sample_neighbors_from_mc(res=self.grid_res_for_lap, shape_code=shape_latent_code, bound_min=self.bound_min, bound_max=self.bound_max)
                points_sphere = points_surface + self.deform_net(points_surface, shape_latent_code)
                points_sdf = self.implicit_net(points_sphere)
                grad_temp = self.get_gradient(points_sphere, points_sdf)
                grad_temp = grad_temp / grad_temp.norm(2, -1, keepdim=True)
                
                laplace_loss_tan, laplace_loss_nor = self.get_laplace_loss_from_neighbors(points_sphere, grad_temp, points_sphere[neighbors_indices], lap_weights)
                output['laplace_loss_tan'] = laplace_loss_tan #[n,3]
                output['laplace_loss_nor'] = laplace_loss_nor

        else:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            pts_norm = points_flat.norm(2, -1, keepdim=True)
            inside_sphere = (pts_norm < 3.0).float().detach() # filter outside sphere 'r > 3
            normals = normals * inside_sphere
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            if self.white_bkgd:
                # apply bkgd assumption on normal
                acc_map = torch.sum(weights, -1)
                normal_map = normal_map + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)
            output['normal_map'] = normal_map

        return output
    
        
    def compute_exp_weights(self, pts_on_surface, neighbors_on_surface, edge=1.0):
        dist = ((pts_on_surface[:,None,:]-neighbors_on_surface)**2).sum(dim=-1).sqrt()
        edge = dist.mean(dim=-1, keepdim=True)
        lap_weights = torch.exp(-dist/edge) #[n,m]
        lap_weights = lap_weights / lap_weights.norm(1, -1, keepdim=True)
        return lap_weights

    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        # print('threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles
    
    def extract_surface_points(self, bound_min, bound_max, res=64, level=0, shape_code=None):
        vertices, triangles = \
        self.extract_geometry(bound_min, bound_max, resolution=res, threshold=level,\
                          query_func=lambda x: self.get_sdf_vals(x.to(shape_code.device), shape_code))
        tmp_mesh = trimesh.Trimesh(vertices, triangles)
        avg_edge = np.array(tmp_mesh.edges_unique_length).mean()
        return vertices, avg_edge
    
    def sample_surface_points(self, res=128, shape_code=None, bound_min=[-1.5, -1.5 -1.5], bound_max=[1.5, 1.5, 1.5]):
        bound_min = torch.tensor(bound_min).float().cuda()
        bound_max = torch.tensor(bound_max).float().cuda()
        points_space, avg_edge = self.extract_surface_points(bound_min, bound_max, res=res, shape_code=shape_code)
        points_surface = torch.from_numpy(points_space).cuda().float()
        points_surface = points_surface[points_surface.norm(2,-1)<=1.3]
        return points_surface, avg_edge
    
    def get_neighbors_from_surface_points(self, points_surface, num_neighbors=6):
        dist = torch.norm((points_surface[None, :, :] - points_surface[:,None,:]), dim=-1, p=2)
        _, indices = torch.topk(dist, num_neighbors + 1, largest=False, dim=1)
        neighbor_indices = indices[:, 1:] #[n,m]
        # neighbors = points_surface[neighbor_indices]
        return neighbor_indices
    
    def sample_neighbors_from_mc(self, res=128, shape_code=None, bound_min=[-1.5, -1.5 -1.5], bound_max=[1.5, 1.5, 1.5]):
        with torch.no_grad():
            points_surface, avg_edge = self.sample_surface_points(res, shape_code, bound_min=bound_min, bound_max=bound_max)
        neighbors_indices = self.get_neighbors_from_surface_points(points_surface)
        lap_weights = self.compute_exp_weights(points_surface, points_surface[neighbors_indices], edge=avg_edge)
        return points_surface, neighbors_indices, lap_weights
    
    def get_sdf_grad_output(self, pts, shape_code=None):
        assert shape_code is not None, print('shapecode is none when get outputs in facevr')
        pts.requires_grad_(True)
        pts_on_sphere = pts + self.deform_net(pts, shape_code)
        sdf = self.implicit_net(pts_on_sphere)
        grad = self.get_gradient(pts, sdf)
        return sdf, grad

    def get_vcolor_from_texmap(self, pts_on_sphere):
        vcolors = sample_cubemap(self.tex_images_ref, pts_on_sphere)
        return vcolors
    
    def transfer_sphere_coords_to_tex_coords(self, pts_on_sphere):
        # this uv is not randomly generated, not the cubemap resutls.
        self.W = 512
        self.H = 512
        x = pts_on_sphere[:,:1]
        y = pts_on_sphere[:,1:2]
        z = pts_on_sphere[:,2:3]
        theta = torch.arctan2(y, z)
        xy_norm = torch.sqrt(x*x + y*y)
        phi = torch.arctan2(z, xy_norm)
        uv = torch.cat([theta,phi], -1)

        # u = (theta + torch.pi) / (2 * torch.pi) * self.W
        # v = (1 - (torch.pi / 2 - phi) / torch.pi) * self.H
        # uv = torch.cat([u,v], -1)
        return uv

    
    def update_embed_fn(self, alpha=1000.):
        self.deform_net.update_embed_fn(alpha)
        if self.use_inverse_flag:
            self.deform_inv_net.update_embed_fn(alpha)
        self.tex_network.update_embed_fn(alpha)
        if self.shade_network is not None:
            self.shade_network.update_embed_fn(alpha)

    def get_embedding(self, shape_idx=[-1],color_idx=[-1]):
        shape_embedding = self.id_shape_embedding
        all_ids = torch.arange(0, self.num_id).cuda()
        if shape_idx[0] == -1:
            shape_code = shape_embedding(all_ids)
        else:
            if len(shape_idx.shape) > 1:
                shape_idx = shape_idx.squeeze(0)
            shape_code = shape_embedding(shape_idx)
        color_code = None
        color_embedding = self.id_color_embedding
        if color_idx[0] == -1:
            color_code = color_embedding(all_ids)
        else:
            if len(color_idx.shape) > 1:
                color_idx = color_idx.squeeze(0)
            color_code = color_embedding(color_idx)
        embeddings = {
            'shape_code' : shape_code,
            'color_code' : color_code,
        }
        return embeddings
    
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

    def get_laplace_loss(self, x, normal, neigh_points_indices, lap_weights):
        # for the simple implement, weight=1/m
        # x is p_prim, the points on the ball, normal is the normal of the point on the ball
        # neigh_points_indices [n, m]
        # only success in case that x.shape equal to loaded dataset.shape
        x = x[:neigh_points_indices.shape[0], :]
        normal = normal[:neigh_points_indices.shape[0], :]
        indict_non_index = x.shape[0]
        padding = torch.zeros((1, 3)).to(x.device)
        new_x = torch.cat((x, padding), dim=0)

        neighborhoods = new_x[neigh_points_indices] #[n, m, 3]
        weight_norm = lap_weights[:,:,None]
        weight_neighbors = (weight_norm * neighborhoods).sum(dim=1)  #[n, 3]
        differences = weight_neighbors - x
   
        normal_axis_vec = torch.sum(differences*normal, dim=1, keepdim=True) * normal
        tanget_axis_vec = differences - normal_axis_vec
        return tanget_axis_vec, differences
    
    def get_laplace_loss_from_neighbors(self, x, normal, neighborhoods, lap_weights):
        # x is p_prim, the points on the sphere, normal is the normal of the point on the sphere
        weight_norm = lap_weights[:,:,None]
        weight_neighbors = (weight_norm * neighborhoods).sum(dim=1)  #[n, 3]
        differences = weight_neighbors - x
        normal_axis_vec = torch.sum(differences*normal, dim=1, keepdim=True) * normal
        tanget_axis_vec = differences - normal_axis_vec
        return tanget_axis_vec, differences

    def set_plot_template(self, plot_template):
        self.plot_template = plot_template

    def nearest_neighbors_indices(self, x, mesh_points):
        # Compute pairwise distances between points in A and B
        dist_matrix = torch.cdist(x, mesh_points)  # Shape: (m, n)
        # Find the index of the nearest point in B for each point in A
        _, indices = torch.min(dist_matrix, dim=1)  # Shape: (m,)
        return indices

    def get_texcoords_given_mesh(self, x, mesh_points, mesh_texcoords):
        indices = self.nearest_neighbors_indices(x, mesh_points)
        texcoords = mesh_texcoords[indices]
        return texcoords

    def get_sdf_vals(self, x, shape_code=None):
        if not self.plot_template:
            d2t_offset = self.deform_net(x, shape_latent_code=shape_code)
            points_temp = x + d2t_offset
            sdf = self.implicit_net(points_temp)
        else:
            sdf = self.implicit_net(x)
        return sdf
    
    def get_vertex_colors(self, x, dirs=None, shape_code=None, color_code=None):
        x.requires_grad_(True)
        d2t_offset = self.deform_net(x,shape_latent_code=shape_code)
        points_temp = x + d2t_offset
        sdf = self.implicit_net(points_temp)
        gradients = self.get_gradient(x, sdf)
        if self.ref_to_tex_path is not None:
            tex_flat = self.get_vcolor_from_texmap(points_temp)
        else:
            color_latent_code_repeat = color_code.repeat(points_temp.shape[0],1)
            tex_flat = self.tex_network(points=points_temp, normals=gradients, feature_vectors=color_latent_code_repeat)
        uv_flat = self.transfer_sphere_coords_to_tex_coords(points_temp) # its randomly generated, not truely used
        if False and dirs is not None:
            # return shading affection
            dirs = dirs.repeat(points_temp.shape[0],1)
            shade_view_flat = self.shade_network(points=points_temp, normals=gradients, view_dirs=dirs, feature_vectors=color_latent_code_repeat)
            return torch.exp(shade_view_flat).repeat(1,3) * tex_flat, uv_flat
        return tex_flat, uv_flat
    
    def get_p_normal_from_p_prime(self, p, shape_code=None):
        sdf = self.get_sdf_vals(p, shape_code)
        grad = self.get_gradient(p, sdf)
        return grad

    def get_pprim_from_p(self, x, shape_code=None):
        with torch.no_grad():
            d2t_offset = self.deform_net(x, shape_code)
            points_temp = x + d2t_offset
        return points_temp
    
    def get_pori_from_pprim(self, x, shape_code=None):
        with torch.no_grad():
            t2d_offset = self.deform_inv_net(x, shape_code)
            points_ori = x + t2d_offset
        return points_ori

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        assert torch.isnan(dists).sum() == 0, print('dists met a nan')
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        assert torch.isnan(alpha).sum() == 0, print('alpha met a nan')
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        assert torch.isnan(transmittance).sum() == 0, print('transmittance met a nan')
        weights = alpha * transmittance # probability of the ray hits something here

        return weights
