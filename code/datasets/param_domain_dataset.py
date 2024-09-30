import os
import torch
import numpy as np

from mesh_to_sdf import sample_sdf_near_surface
import trimesh
from sklearn.neighbors import KDTree


class ParamDomainDataset(torch.utils.data.Dataset):
    '''
    neural parameterization under volume rendering framework
    '''
    def __init__(self, **kwargs):
        super(ParamDomainDataset, self).__init__()
        os.environ['PYOPENGL_PLATFORM']='egl'

        self.conf = kwargs['conf']
        self.resample_flag = self.conf.get_bool('resample',default=False)
        self.sample_surface_from_mesh = self.conf.get_bool('sample_surface', default=False)
        self.num_points = self.conf['num_points']
        self.data_dir = self.conf['data_dir']
        self.debug_flag = False

        assert os.path.exists(self.data_dir), "Data directory is empty"

        self.sampling_idx = None
        self.scan_id = self.conf.get_string('scan_id', default='sphere')
        self.cam_scale = self.conf.get_float('cam_scale', default=1.0)
        self.img_res = self.conf['img_res'] # no use for mesh
        self.total_pixels = self.img_res[0] * self.img_res[1]

        print('Dataset load '.format(self.scan_id))

        self.names = []
        self.points_all = []
        self.sdf_all = []
        self.grad_all = []
        self.lap_weights_all = []
        self.neighbor_indices_all = []
    
        instance_path = os.path.join(self.data_dir, self.scan_id)
        assert os.path.exists(instance_path), f"{instance_path} doesnt exist"
        instance_list = os.listdir(instance_path)

        for item in instance_list:
            if '.obj' in item or '.ply' in item:
                file_name = os.path.join(instance_path, item)
                print(f'loading {file_name}...')

                points_all, grad_all, sdf_all = \
                    self.get_points_from_mesh(file_name, 
                                                need_subdivide=False, 
                                                resample=self.resample_flag, 
                                                resample_size=self.num_points)
                self.points_all.append(torch.from_numpy(np.array(points_all)).float())
                self.grad_all.append(torch.from_numpy(np.array(grad_all)).float())
                self.sdf_all.append(torch.from_numpy(np.array(sdf_all)).float())
                break # load one object
        self.ids = torch.from_numpy(np.array([0])) # [1,1,0,0,0,1,2]

    def __len__(self):
        return len(self.points_all)

    def __getitem__(self, idx):
        sample = {
            "id": self.ids[idx],
            "points": self.points_all[idx],
        }
        ground_truth = {
            "sdf": self.sdf_all[idx],
            "grad": self.grad_all[idx]
        }
        return idx, sample, ground_truth

    def compute_normals_via_mesh(self, verts, tris):
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
    
    def get_points_from_mesh(self, mesh_path, need_subdivide=False, resample=False, resample_size=5000, need_rotate=False):
        # mesh: trimesh.load(path)
        # return p [n, 3], neighbor(p) [n, m]. m is the maximum num of the neighborhood, -1 indicates none of neighbor id
        face_mesh = trimesh.load(mesh_path)
        if need_subdivide:
            face_mesh = face_mesh.subdivide().subdivide()
        points_all = face_mesh.vertices
        grad_all = self.compute_normals_via_mesh(points_all, face_mesh.faces)
        sdf_all = np.zeros((points_all.shape[0], 1))

        if resample:
            # resampling the points from the given mesh
            if not self.sample_surface_from_mesh:
                print('sample points sdf near surface')
                points_all_sample, sdf_all_sample, grad_all_sample = sample_sdf_near_surface(face_mesh,resample_size,return_gradients=True) # dont normalize mesh
                sdf_all_sample = sdf_all_sample.reshape(-1, 1)
            else:
                print('sample points on surface')
                points_all_sample, face_index = trimesh.sample.sample_surface_even(face_mesh, resample_size, radius=None) # dont normalize mesh
                sdf_all_sample = np.zeros((points_all_sample.shape[0],1))
                grad_all_sample = face_mesh.face_normals[face_index]
            return points_all_sample, grad_all_sample, sdf_all_sample
        
        return points_all, grad_all, sdf_all

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
    
    def change_sampling_idx(self, sampling_size, semantic_pred=None, semantic_gt=None, rgb_pred=None, rgb_gt=None, w=0):
        self.sampling_size = sampling_size
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
