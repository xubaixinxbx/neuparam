import numpy as np
import torch
import torchvision
import trimesh
from PIL import Image
import mcubes


from utils import rend_util
from utils.color_corres import get_uv_given_mesh
from utils.plot_sdf import plot_cuts_iso

def extract_fields(bound_min, bound_max, resolution, query_func):
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

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def sdf_func(x, shape_code, net):
    return net.get_sdf_vals(x,shape_code)

def plot_mesh(net, shape_idx, path, epoch, resolution, grid_boundary, level=0, cam_scale=1.0, gt_mesh_path=None):
    print('level=',level)

    # plot surface
    color_idx = shape_idx
    code_embeddings = net.get_embedding(shape_idx=shape_idx,color_idx=color_idx)
    shape_code = code_embeddings['shape_code']
    device = shape_code.device if shape_code is not None else net.device
    print(f'shape_code={shape_idx.item()}, color_code={color_idx.item()}')

    bound_min = torch.tensor([grid_boundary[0],grid_boundary[0],grid_boundary[0]], dtype=torch.float32)
    bound_max = torch.tensor([grid_boundary[1],grid_boundary[1],grid_boundary[1]], dtype=torch.float32)
    vertices, triangles = \
        extract_geometry(bound_min, bound_max, resolution=resolution, threshold=level,\
                          query_func=lambda x: net.get_sdf_vals(x.to(device), shape_code))

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export('{0}/{1}.ply'.format(path, str(epoch)+f'_{shape_idx.item()}'), 'ply')


def plot_parameterization(net, shape_idx, path, epoch, export_uvtex=False, uvtex_res=128, dirs_shading=[0,0,-1]):
    # dirs_shading: the directions from which shading is viewed
    color_idx = shape_idx
    save_path =  '{0}/surface_{1}.obj'.format(path, str(epoch)+f'_{shape_idx.item()}')
    mesh_path = '{0}/{1}.ply'.format(path, str(epoch)+f'_{shape_idx.item()}')
    face_mesh = trimesh.load(mesh_path)
    dirs_sample = torch.from_numpy(np.array(dirs_shading)).float().cuda().reshape(1, 3)
    get_uv_given_mesh(net, shape_idx, color_idx, face_mesh, save_path, export_uvtex=export_uvtex, uvtex_res=uvtex_res, dirs_given=dirs_sample)

def plot(net, indices, plot_data, path, epoch, img_res, plot_nimgs, resolution, grid_boundary,\
        cam_scale=1.0, writer=None, uvtex_res=128, plot_template = False, \
        level=0, only_image=False, export_uvtex=False, cam_id='0', dirs_shading=[0,0,-1]):
    rgb, normal = None, None
    if plot_data is not None:
        if 'pose' in plot_data:
            cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])
            rgb=plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, str(epoch)+'_'+str(plot_data['id'].item())+'_'+cam_id, plot_nimgs, img_res, writer,render_only=True)
            normal=plot_normal_maps(plot_data['normal_map'], path, str(epoch)+'_'+str(plot_data['id'].item())+'_'+cam_id, plot_nimgs, img_res, writer)
        
        if only_image:
            return rgb,normal
    # extract mesh
    plot_mesh(net, indices, path, epoch, resolution, grid_boundary, level, cam_scale, gt_mesh_path=None)
    # extract parameterization results
    plot_parameterization(net, indices, path=path, epoch=epoch, export_uvtex=export_uvtex, uvtex_res=uvtex_res, dirs_shading=dirs_shading)
    

def plot_normal_maps(normal_maps, path, epoch, plot_nrow, img_res, writer=None, no_save=False):
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    if writer is not None:
        writer.add_image("normal",tensor.transpose(2,0,1),int(epoch.split('_')[0]))
    img = Image.fromarray(tensor)    
    if not no_save:
        img.save('{0}/normal_{1}.png'.format(path, epoch))
    return img

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, writer=None, render_only=False, no_save=False):
    if not render_only:
        ground_true = ground_true.cuda()
        output = torch.cat((rgb_points, ground_true), dim=0)
    else:
        output = rgb_points
    output_plot = lin2img(output, img_res)

    tensor = torchvision.utils.make_grid(output_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)
    if writer is not None:
        writer.add_image('image',tensor.transpose(2,0,1),int(epoch.split('-')[0]))
    img = Image.fromarray(tensor)
    if not no_save:
        img.save('{0}/rendering_{1}.png'.format(path, epoch))
    return img


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
