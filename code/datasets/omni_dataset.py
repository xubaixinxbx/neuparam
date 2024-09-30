import os
import torch
import numpy as np

import utils.general as utils
import json
import cv2 

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                  **kwargs
                 ):
        super(SceneDataset, self).__init__()
        self.conf = kwargs['conf']
        self.sampling_idx = None
        self.scan_id = int(self.conf.get_string('scan_id', default='-1'))
        self.cam_scale = self.conf.get_float('cam_scale', default=1)
        self.img_res = self.conf['img_res']
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.data_dir = self.conf['data_dir']
        class_name = self.data_dir.split('/')[-1]
        self.instance_dir = os.path.join(self.data_dir, '{}_{:03}/render'.format(class_name,self.scan_id))

        assert os.path.exists(self.instance_dir), "Data directory is empty"


        image_dir = '{0}/images'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.names = []
        for i in image_paths:
            self.names.append(i.split('/')[-1][:-4])
            
        self.n_images = len(image_paths)

        self.cam_file = '{0}/transforms.json'.format(self.instance_dir)
        

        imgs, poses, hwf, i_split, img_files = self.load_blender_data(
            basedir = self.instance_dir, half_res = False, test_ratio = 0.125
        )
        # train with all images as default

        self.rgb_images_all = torch.from_numpy(imgs).float()
        self.pose_all = torch.from_numpy(poses).float()
        k = np.eye(4)
        k[0, 0] = k[1, 1] = hwf[2]
        k[0, 2] = hwf[0] / 2.0
        k[1, 2] = hwf[1] / 2.0
        self.intrinsic_all = torch.from_numpy(np.repeat(k.reshape(1, 4, 4), axis=0, repeats=self.pose_all.shape[0])).float()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsic_all[idx],
            "pose": self.pose_all[idx],
            "id": torch.zeros(1)[0].long(),
        }

        ground_truth = {
            "rgb": self.rgb_images_all[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images_all[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

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

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def load_blender_data(self, basedir, half_res=False, test_ratio=0.125):
        with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)

        counts = [0]
        imgs = []
        img_files = []
        poses = []

        for frame in meta['frames']:
            fname = os.path.join(basedir, 'images', frame['file_path'].split('/')[-1] + '.png')
            img_files.append(fname)
            img = np.array(cv2.imread(fname, cv2.IMREAD_UNCHANGED))
            if img.shape[-1] > 3:
                alpha = img[:,:,3:] / 255.0
                img = img[:,:,:3] * alpha  + (1 - alpha) * np.array([0,0,0]).reshape(1,1,3)

            img = img[..., ::-1]
            H, W = img.shape[:2]
            imgs.append(img.reshape(-1, 3))
            pose = np.array(frame['transform_matrix'])
            pose[:,1:3] *= -1
            poses.append(pose)
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        print(imgs.shape, poses.shape)

        n_images = len(imgs)
        freq_test = int(1/test_ratio)
        i_val = i_test = np.arange(0, n_images, freq_test)
        i_train = np.asarray(list(set(np.arange(n_images).tolist())-set(i_test.tolist())))
        i_split = [i_train, i_val, i_test]
        print('TRAIN views are', i_train)
        print('VAL views are', i_val)
        print('TEST views are', i_test)

        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        return imgs, poses, [H, W, focal], i_split, img_files