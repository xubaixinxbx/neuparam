import os 
import trimesh
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh
from scipy.spatial import KDTree


def compute_normals_via_mesh(verts, tris):
    #  verts: mesh.vertices, mesh: trimesh
    #  tris: mesh.faces
    normals = np.zeros(verts.shape)
    tri_verts = verts[tris]
    n0 = np.cross(tri_verts[::, 1] - tri_verts[::, 0], tri_verts[::, 2] - tri_verts[::, 0])
    n0 = n0 / np.linalg.norm(n0, axis=1)[:, np.newaxis]
    for i in range(tris.shape[0]):
        normals[tris[i, 0]] += n0[i]
        normals[tris[i, 1]] += n0[i]
        normals[tris[i, 2]] += n0[i]
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals

def find_pprim_given_p(p, net, shape_code):
    with torch.no_grad():
        return net.get_pprim_from_p(torch.from_numpy(p).float().cuda().reshape(p.shape[0],3), shape_code)

def transfer_sphere_to_face(net, shape_code, sphere_mesh, path):
    pts_on_sphere = torch.from_numpy(sphere_mesh.vertices).float().cuda()
    pts_on_face = net.get_pori_from_pprim(pts_on_sphere, shape_code)

    face_mesh = trimesh.Trimesh(vertices=pts_on_face.detach().cpu().numpy(), faces=sphere_mesh.faces, vertex_colors=sphere_mesh.visual.vertex_colors)
    face_mesh.export(path.replace('surface','surface_from_sphere'))

def transfer_face_to_sphere(net, shape_code, face_mesh, path):
    vertex_all = face_mesh.vertices
    grad_all = compute_normals_via_mesh(vertex_all, face_mesh.faces)
    face_all = face_mesh.faces
    
    pprim = find_pprim_given_p(vertex_all, net, shape_code).detach().cpu().numpy()
    
    sphere_mesh = trimesh.Trimesh(pprim, face_all, vertex_normals=grad_all, vertex_colors=face_mesh.visual.vertex_colors)
    sphere_mesh.export(path.replace('surface', 'sphere'),'obj')

    # transfer sphere to face
    transfer_sphere_to_face(net, shape_code, sphere_mesh, path)

    return sphere_mesh

def sample_cubemap(cubemap, xyz):
    assert len(cubemap.shape) == 4
    assert cubemap.shape[0] == 6
    assert cubemap.shape[1] == cubemap.shape[2]
    assert xyz.shape[-1] == 3

    result = torch.zeros(xyz.shape[:-1] + (cubemap.shape[-1],)).float().to(xyz.device)

    x, y, z = xyz.unbind(-1)

    absX = x.abs()
    absY = y.abs()
    absZ = z.abs()

    isXPositive = x > 0
    isYPositive = y > 0
    isZPositive = z > 0

    maps = cubemap.unbind(0)
    masks = [
        isXPositive * (absX >= absY) * (absX >= absZ),
        isXPositive.logical_not() * (absX >= absY) * (absX >= absZ),
        isYPositive * (absY >= absX) * (absY >= absZ),
        isYPositive.logical_not() * (absY >= absX) * (absY >= absZ),
        isZPositive * (absZ >= absX) * (absZ >= absY),
        isZPositive.logical_not() * (absZ >= absX) * (absZ >= absY),
    ]

    uvs = []

    uc = -z[masks[0]] / absX[masks[0]]
    vc = y[masks[0]] / absX[masks[0]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = z[masks[1]] / absX[masks[1]]
    vc = y[masks[1]] / absX[masks[1]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[2]] / absY[masks[2]]
    vc = -z[masks[2]] / absY[masks[2]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[3]] / absY[masks[3]]
    vc = z[masks[3]] / absY[masks[3]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[4]] / absZ[masks[4]]
    vc = y[masks[4]] / absZ[masks[4]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = -x[masks[5]] / absZ[masks[5]]
    vc = y[masks[5]] / absZ[masks[5]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    for texture, mask, uv in zip(maps, masks, uvs):
        result[mask] = (
            F.grid_sample(
                texture.permute(2, 0, 1)[None],
                uv.view((1, -1, 1, 2)),
                padding_mode="border",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .view(uv.shape[:-1] + (texture.shape[-1],))
        )

    return result

def load_cube_from_single_texture(filename, rotate=True):
    img = np.array(Image.open(filename)) / 255.0
    # img = np.concatenate([img[:,:,None], img[:,:,None], img[:,:,None]], -1)
    assert img.shape[0] * 4 == img.shape[1] * 3
    res = img.shape[0] // 3
    if rotate:
        cube = [
            img[res : 2 * res, :res][::-1],
            img[res : 2 * res, 2 * res : 3 * res][::-1],
            img[:res, res : 2 * res][:, ::-1],
            img[2 * res : 3 * res, res : 2 * res][:, ::-1],
            img[res : 2 * res, 3 * res :][::-1],
            img[res : 2 * res, res : 2 * res][::-1],
        ]
    else:
        cube = [
            img[res : 2 * res, 2 * res : 3 * res][::-1],
            img[res : 2 * res, :res][::-1],
            img[:res, res : 2 * res][::-1],
            img[2 * res : 3 * res, res : 2 * res][::-1],
            img[res : 2 * res, res : 2 * res][::-1],
            img[res : 2 * res, 3 * res :][::-1],
        ]

    return cube

def load_cubemap(imgs):
    assert len(imgs) == 6
    cubemaps = np.array([np.array(Image.open(img).resize((1024,1024)))[::-1] / 255.0 for img in imgs])
    return cubemaps

def generate_grid(dim, resolution):
    grid = np.stack(
        np.meshgrid(*([np.arange(resolution)] * dim), indexing="ij"), axis=-1
    )
    grid = (2 * grid + 1) / resolution - 1
    return grid

def convert_cube_uv_to_xyz(index, uvc, need_normalized=True):
    assert uvc.shape[-1] == 2
    # import pdb; pdb.set_trace()
    vc, uc = uvc.unbind(-1)
    if index == 0:
        x = torch.ones_like(uc).to(uc.device)
        y = vc
        z = -uc
    elif index == 1:
        x = -torch.ones_like(uc).to(uc.device)
        y = vc
        z = uc
    elif index == 2:
        x = uc
        y = torch.ones_like(uc).to(uc.device)
        z = -vc
    elif index == 3:
        x = uc
        y = -torch.ones_like(uc).to(uc.device)
        z = vc
    elif index == 4:
        x = uc
        y = vc
        z = torch.ones_like(uc).to(uc.device)
    elif index == 5:
        x = -uc
        y = vc
        z = -torch.ones_like(uc).to(uc.device)
    else:
        raise ValueError(f"invalid index {index}")
    if need_normalized:
        return F.normalize(torch.stack([x, y, z], axis=-1), dim=-1)
    else:
        return torch.stack([x,y,z], axis=-1)

def merge_cube_to_single_texture(cube, flip=True, rotate=True):
    """
    cube: (6,res,res,c)
    """
    assert cube.shape[0] == 6
    assert cube.shape[1] == cube.shape[2]
    res = cube.shape[1]
    result = torch.ones((3 * res, 4 * res, cube.shape[-1]))

    if flip:
        cube = cube.flip(1)
    if rotate:
        result[res : 2 * res, :res] = cube[0]
        result[res : 2 * res, res : 2 * res] = cube[5]
        result[res : 2 * res, 2 * res : 3 * res] = cube[1]
        result[res : 2 * res, 3 * res :] = cube[4]
        result[:res, res : 2 * res] = cube[2].flip(0, 1)
        result[2 * res : 3 * res, res : 2 * res] = cube[3].flip(0, 1)
    else:
        result[res : 2 * res, :res] = cube[1]
        result[res : 2 * res, res : 2 * res] = cube[4]
        result[res : 2 * res, 2 * res : 3 * res] = cube[0]
        result[res : 2 * res, 3 * res :] = cube[5]
        result[:res, res : 2 * res] = cube[2]
        result[2 * res : 3 * res, res : 2 * res] = cube[3]

    return result

# def interpolating_vcolor_from_pts(query_pts, near_pts, near_pts_vcolor):
def points_to_barycentric(triangles, pts):
    # triangles [n,3,3], pts[n,3], torch.Tensor
    def diagonal_dot(a, b):
        # import pdb; pdb.set_trace()
        return torch.mm(a * b, torch.tensor([1.0] * a.shape[1]).reshape(3,1).float().to(a.device)).squeeze()
    def method_cramer():
        dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
        dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
        dot02 = diagonal_dot(edge_vectors[:, 0], w)
        dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
        dot12 = diagonal_dot(edge_vectors[:, 1], w)

        inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01)

        barycentric = torch.zeros((len(triangles), 3)).float().to(pts.device)
        barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
        barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = pts - triangles[:, 0].reshape((-1, 3))
    return method_cramer()

def interpolating_vcolor_on_sphere(pts_given_sphere, sphere_transferred_mesh):
    pts_given_sphere = (pts_given_sphere).detach().cpu().numpy()
    proximity = trimesh.proximity.ProximityQuery(sphere_transferred_mesh)
    closest, distance, triangle_id = proximity.on_surface(pts_given_sphere)
    closest_tri_id = sphere_transferred_mesh.faces[triangle_id] #[n,3]
    closest_tri_points = sphere_transferred_mesh.vertices[closest_tri_id] #[n,3,3]
    barycentric = trimesh.triangles.points_to_barycentric(closest_tri_points, pts_given_sphere)
    if hasattr(sphere_transferred_mesh.visual,'uv') and sphere_transferred_mesh.visual.uv is not None:
        color_visual = sphere_transferred_mesh.visual.to_color()
    else:
        color_visual = sphere_transferred_mesh.visual
    vcolor = color_visual.vertex_colors[closest_tri_id] #[n,3,4]
    vcolor = (barycentric[:, :, None] * vcolor).sum(axis=1)

    return torch.from_numpy(vcolor[:,:3]).float().cuda() / 255.0

def export_cube_textures(resolution=512, net=None, dirs_given=None, shape_code=None, color_code=None, save_dir=None, idx=-1, epoch='0', sphere_color_mesh=None):
    grid = torch.tensor(generate_grid(2, resolution)).float().cuda()
    textures = []
    for index in range(6):
        xyz = convert_cube_uv_to_xyz(index, grid, need_normalized=True)
        xyz = xyz.reshape(-1, 3) #(512*512, 3)
        proximity = trimesh.proximity.ProximityQuery(sphere_color_mesh)
        closest, distance, triangle_id = proximity.on_surface(xyz.data.cpu().numpy())
        xyz_normals= torch.from_numpy(sphere_color_mesh.face_normals[triangle_id]).float().cuda()
        xyz_normals_all = xyz_normals.split(1024*128)
        xyz_all = xyz.split(1024*128)
        vcolor_list = []
        for xyz_batch, xyz_normals_batch in zip(xyz_all, xyz_normals_all):
            vcolor_batch = interpolating_vcolor_on_sphere(xyz_batch, sphere_color_mesh)
            vcolor_list.append(vcolor_batch)
        vcolor = torch.concat(vcolor_list, 0)
        vcolor = vcolor.reshape(resolution, resolution, 3)
        textures.append(vcolor)
    textures = torch.stack(textures, dim=0)
    textures = merge_cube_to_single_texture(textures)
    textures = textures.clamp(0, 1).data.cpu().numpy()
    Image.fromarray((textures * 255).astype(np.uint8)).save(
        os.path.join(save_dir, "texture_{}_{}.png".format(idx, epoch))
    )

def get_uv_given_mesh(net, shape_idx, color_idx, face_mesh, path, export_uvtex=False, uvtex_res=128, dirs_given=None):
    #path: ...../surface.obj
    code_embeddings = net.get_embedding(shape_idx=shape_idx,color_idx=color_idx)
    shape_code = code_embeddings['shape_code']
    color_code = code_embeddings['color_code']

    pts_on_face = torch.from_numpy(face_mesh.vertices).float().cuda().split(1024*64)
    tex = []
    uv = []
    for pts_on_face_batch in pts_on_face:
        tex_batch, uv_batch= net.get_vertex_colors(x=pts_on_face_batch, dirs=dirs_given,
                        shape_code=shape_code, color_code=color_code)
        tex_batch = torch.clamp(tex_batch, 0, 1)
        tex.append(tex_batch.detach().cpu().numpy())
        uv.append(uv_batch.detach().cpu().numpy())
    tex = np.concatenate(tex, 0)
    uv = np.concatenate(uv, 0)
    face_tex_mesh = trimesh.Trimesh(vertices=face_mesh.vertices,faces=face_mesh.faces,
                                    vertex_colors=tex)
    face_mesh = face_tex_mesh
    face_mesh.export(path,'obj')
    print('uv plot done')
    
    sphere_mesh = transfer_face_to_sphere(net, shape_code, face_mesh, path)
    print('face transfer sphere done')
    if export_uvtex:
        sphere_color_mesh = trimesh.Trimesh(vertices=sphere_mesh.vertices, faces=sphere_mesh.faces, vertex_normals=sphere_mesh.vertex_normals, vertex_colors=face_mesh.visual.vertex_colors)
        # extract flatten texture
        cur_path = os.path.dirname(path)
        save_path = os.path.join(cur_path)
        export_cube_textures(resolution=uvtex_res, net=net, dirs_given=dirs_given, shape_code=shape_code, 
                                color_code=color_code, save_dir=save_path, 
                                idx=color_idx.item(), epoch=path.split('_')[-2], sphere_color_mesh=sphere_color_mesh)

def sample_points_on_parametric_and_transfer_to_original(parametric_path, save_path, net, shape_code):
    param_surface = trimesh.load(parametric_path)
    points_all_sample, face_index = trimesh.sample.sample_surface_even(param_surface, 10000, radius=None)
    pts_on_sphere = torch.from_numpy(points_all_sample).float().cuda()
    pts_on_face = net.get_pori_from_pprim(pts_on_sphere, shape_code)

    face_mesh = trimesh.Trimesh(vertices=pts_on_face.detach().cpu().numpy())
    face_mesh.export(save_path)

if __name__ == '__main__':
    pass
