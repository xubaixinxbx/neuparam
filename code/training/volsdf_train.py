import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
from shutil import copyfile
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from tensorboardX import SummaryWriter
# sys.path.append('/home/bxxu/data1/miniconda3/envs/nerf/lib/python3.9/site-packages')

class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        f = open(kwargs['conf'])
        conf_text = f.read()
        conf_text = conf_text.replace('SCAN_ID', str(kwargs['scan_id']))
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf_path = kwargs['conf']
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        lustre_exp_path = self.conf.get_string('train.root_path')

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(lustre_exp_path,kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join(lustre_exp_path,kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
            if self.conf.get_string('train.assign_checkpnts_dir', default=None) is not None:
                is_continue = True
                timestamp = kwargs['timestamp']
            print("loading model from timestamp={}".format(timestamp))
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']
        
        utils.mkdir_ifnotexists(os.path.join(lustre_exp_path,self.exps_folder_name))
        self.expdir = os.path.join(lustre_exp_path, self.exps_folder_name, self.expname)
        if '/' in self.expname:
            utils.mkdir_ifnotexists(os.path.join(lustre_exp_path, self.exps_folder_name, self.expname.split('/')[0]))
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)
        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        # set_up tensorboard path
        self.tb_path = os.path.join(self.expdir, self.timestamp, "tb_path") 
        
        self.file_backup()
        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)
        print('shell command : {0}'.format(' '.join(sys.argv)))
        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=dataset_conf)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=0,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        self.lr = self.conf.get_float('train.learning_rate')

        # stage 2 we freeze the parametric domain's params
        if self.conf.get_bool('train.freeze_params', default=False):
            print('----set parametric domain params grad as False----')
            for name, param in self.model.named_parameters():
                if 'implicit_net' in name:
                    param.requires_grad = False
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
            else:
                print('param_not_to_update:',name)
        self.optimizer = torch.optim.Adam(params_to_update, lr=self.lr)

        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            assign_checkpnts_dir = self.conf.get_string('train.assign_checkpnts_dir', default=None)
            if assign_checkpnts_dir is not None and os.path.exists(assign_checkpnts_dir):
                old_checkpnts_dir = assign_checkpnts_dir
            else:
                old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            
            self.start_epoch = saved_model_state['epoch']
            # embedding dim in saved_model_state not matched in stage2(1 vs 30)
            # check if need to remove the embedding in saved model state
            new_model_state_dict = saved_model_state['model_state_dict'].copy()
            num_identities_in_model = 0
            for name, values in self.model.named_parameters():
                if 'id_shape_embedding' in name:
                    num_identities_in_model = values.shape[0]
                    break
            for name in new_model_state_dict.keys():
                if 'id_shape_embedding' in name and new_model_state_dict[name].shape[0] != num_identities_in_model:
                    # saved_model_state['model_state_dict'].pop(name)
                    del saved_model_state['model_state_dict'][name]
                    print(f'del module param from ckpt: {name}')
                    break
            missing_keys, unexpected_keys = self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            print('when loading model, missing keys:',missing_keys)
            print('when loading model, unexpected keys:',unexpected_keys)
            print('loading model from: {} checkpoint={}, start_epoch={} '.format(old_checkpnts_dir, kwargs['checkpoint'], self.start_epoch))

            if len(missing_keys) == 0:
                # load previous optimizer
                data = torch.load(
                    os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
                self.optimizer.load_state_dict(data["optimizer_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.scheduler.load_state_dict(data["scheduler_state_dict"])
            else:
                print('when loading previous optimizer, params unmatched, use initial optimizer')
        
        self.total_pixels = self.train_dataset.total_pixels
        self.num_pixels = self.conf.get_int('train.num_pixels') # batch_rays        
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.plot_conf = self.conf.get_config('plot')
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=1000)
        self.img_res = self.train_dataset.img_res
        self.is_parametric_domain = self.conf.get_bool('plot.is_parametric_domain',default=False)
        self.color_id = kwargs.get('color_id', -1)

    def get_color_embedding_shading_net(self, ckpt_path):
        saved_model_state = torch.load(os.path.join(ckpt_path, "ModelParameters/latest.pth"))
        color_id = 0
        self.start_epoch = saved_model_state['epoch']
        color_embedding_dict = {}
        shade_embedding_dict = {}
        for name, values in self.model.named_parameters():
            if 'id_color_embedding' in name:
                color_embedding_dict[name] = values
            if 'shade_network' in name:
                shade_embedding_dict[name] = values
        
        missing_keys, unexpected_keys = self.model.load_state_dict(shade_embedding_dict, strict=False)
        print('when loading shading model, missing keys:',missing_keys)
        print('when loading shading model, unexpected keys:',unexpected_keys)
        import torch.nn as nn
        id_color_embedding = nn.Embedding(num_embeddings=3, embedding_dim=128)
        id_color_embedding.load_state_dict(shade_embedding_dict,strict=False)
        print('when loading color_embedding model, missing keys:',missing_keys)
        print('when loading color_embedding model, unexpected keys:',unexpected_keys)
        color_embedding = id_color_embedding(color_id)
        return color_embedding

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def file_backup(self):
        dir_lis = self.conf['train.recording']
        os.makedirs(os.path.join(self.expdir, self.timestamp, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.expdir, self.timestamp, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.expdir, self.timestamp, 'recording', 'config.conf'))

    def run(self):
        torch.cuda.empty_cache()
        writer = SummaryWriter(logdir=self.tb_path)
        
        print("training...from epoch {0} to {1}".format(self.start_epoch, self.nepochs))
        for epoch in range(self.start_epoch, self.nepochs + 1):
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()
                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                
                if self.color_id != -1:
                    # if color id is specified by user 
                    model_input['color_id'] = torch.zeros_like(model_input['id']) + self.color_id
                for name in model_input:
                    model_input[name] = model_input[name].cuda()

                if not self.is_parametric_domain and self.conf.get_bool('render.render_trainset', default=False):
                    print('render mesh and trainset views')
                    render_trainset_path = os.path.join(self.plots_dir, 'trainset')
                    os.mkdir(render_trainset_path)
                    plt.plot_mesh(
                            self.model,
                            model_input['id'],
                            self.plots_dir,
                            epoch,
                            resolution=self.plot_conf['resolution'],
                            grid_boundary=self.plot_conf['grid_boundary'],
                            level=self.plot_conf['level'],
                            cam_scale=self.conf.get_float('dataset.cam_scale',default=1.0),
                        )
                    poses = self.train_dataset.pose_all
                    intrinsics = self.train_dataset.intrinsic_all
                    rgb_gt = self.train_dataset.rgb_images_all
                    for i in tqdm(range(len(poses))):
                        pose = poses[i]
                        k = intrinsics[i]
                        res = []
                        model_input['pose'] = pose.unsqueeze(0).cuda()
                        model_input['intrinsics'] = k.unsqueeze(0).cuda()
                        ground_truth['rgb'] = rgb_gt[i].unsqueeze(0).cuda()
                        batch_size = ground_truth['rgb'].shape[0]
                        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                        res = []
                        for s in tqdm(split):
                            out = self.model(s)
                            d = {'rgb_values': out['rgb_values'].detach(),
                                'normal_map': out['normal_map'].detach(),
                                }
                            res.append(d)
                        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                        plot_data = self.get_plot_data(model_outputs, model_input['id'], model_input['pose'], ground_truth['rgb'])
                        plt.plot(self.model,
                            indices,
                            plot_data,
                            self.plots_dir,
                            epoch,
                            self.img_res,
                            writer=None,
                            cam_scale=self.conf.get_float('dataset.cam_scale',default=1.0),
                            cam_id=self.train_dataset.names[i],
                            **self.plot_conf
                            )
                    return
                elif self.conf.get_bool('render.render_novel_view', default=False):
                    print("render novel view:")
                    os.system("""rm -rf {0} """.format(os.path.join(self.expdir, self.timestamp)))
                    rgb_imgs = []
                    normal_maps = []
                    num_views, render_path = self.conf.get_int('render.num_views'), self.conf.get_string('render.render_path')
                    start_pose = self.conf.get_int('render.start_pose', default=0)
                    end_pose = self.conf.get_int('render.end_pose', default=1)
                    post_fix = '{}x{}_{}_{}'.format(self.img_res[0], self.img_res[1], num_views, render_path)
                    save_dir = os.path.join(self.plots_dir,'vis',post_fix+str(self.conf.get_int('dataset.scan_id')))
                    os.makedirs(save_dir, exist_ok=True)
                    for i in tqdm(range(num_views)):
                        res = []
                        pose = self.train_dataset.get_novel_pose_between(start_pose, end_pose, i, n_frames=num_views)
                        model_input['pose'] = pose.unsqueeze(0).cuda()
                        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                        torch.cuda.empty_cache()
                        for s in tqdm(split):                 
                            out = self.model(s)
                            res.append({'rgb_values': out['rgb_values'].detach(),
                                        'normal_map': out['normal_map'].detach()})
                        batch_size, num_samples, _ = ground_truth['rgb'].shape
                        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                        rgbs = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
                        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
                        normal_map = (normal_map + 1.) / 2.
                        img = plt.plot_images(
                            rgbs, 
                            None,
                            save_dir, 
                            i, 1, self.img_res, render_only=True)
                        normal = plt.plot_normal_maps(
                            normal_map,
                            save_dir,
                            i,1, self.img_res
                        )
                        rgb_imgs.append(np.asarray(img))
                        normal_maps.append(np.asarray(normal))
                        del out, batch_size, num_samples, model_outputs, res, rgbs
                    for i in range(num_views):
                        rgb_imgs.append(rgb_imgs[num_views-i-1])
                        normal_maps.append(normal_maps[num_views-i-1])
                    w, h = self.img_res[0], self.img_res[1]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_dir = os.path.join(save_dir, 'render')
                    os.makedirs(video_dir, exist_ok=True)
                    writer = cv2.VideoWriter(os.path.join(video_dir,
                                                        '{:0>8d}_{}_rgb.mp4'.format(epoch, post_fix)),
                                            fourcc, 15, (w, h))
                    writer_normal = cv2.VideoWriter(os.path.join(video_dir,
                                                            '{:0>8d}_normal.mp4'.format(epoch)),
                                            fourcc, 15, (w, h))
                    for image in rgb_imgs:
                        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        writer.write(img_bgr)
                    writer.release()
                    for normal in normal_maps:
                        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
                        writer_normal.write(normal_bgr)
                    writer_normal.release()
                    print('finish render only')                   
                    return
                else:
                    if not self.is_parametric_domain:
                        # multiview image as input
                        # plot_data = None
                        batch_size = ground_truth['rgb'].shape[0]
                        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                        res = []
                        for s in tqdm(split):
                            out = self.model(s)
                            d = {'rgb_values': out['rgb_values'].detach(),
                                'normal_map': out['normal_map'].detach(),
                                }
                            res.append(d)
                        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                        plot_data = self.get_plot_data(model_outputs, model_input['id'], model_input['pose'], ground_truth['rgb'])
                        plt.plot(self.model,
                            model_input['id'],
                            plot_data,
                            self.plots_dir,
                            epoch,
                            self.img_res,
                            writer=None,
                            cam_scale=self.conf.get_float('dataset.cam_scale',default=1.0),
                            **self.plot_conf
                            )
                    else:
                        model_outputs = self.model(model_input)
                        plt.plot_mesh(
                                self.model,
                                model_input['id'],
                                self.plots_dir,
                                epoch,
                                resolution=self.plot_conf['resolution'],
                                grid_boundary=self.plot_conf['grid_boundary'],
                                level=self.plot_conf['level'],
                                cam_scale=self.conf.get_float('dataset.cam_scale',default=1.0),
                            )
                  
                    del indices, model_input, ground_truth
                    self.model.train()
                    print('epoch',epoch,'plot over')
                    torch.cuda.empty_cache()
                print(self.expdir, self.timestamp)
            self.train_dataset.change_sampling_idx(self.num_pixels)
            select_print_loss_id = np.random.randint(low=0, high=self.n_batches)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                for name in model_input:
                    model_input[name] = model_input[name].cuda()

                model_outputs = self.model(model_input)
                
                loss_output = self.loss(model_outputs, ground_truth)
                loss = loss_output['loss']
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                if data_index == select_print_loss_id:
                    if self.is_parametric_domain:
                        writer.add_scalar('loss', loss.item(), epoch)
                        writer.add_scalar('sdf_loss', loss_output['sdf_loss'].item(), epoch)
                        writer.add_scalar('grad_loss', loss_output['grad_loss'].item(), epoch)
                        writer.add_scalar('eik_loss', loss_output['eikonal_loss'].item(), epoch)
                        
                        print('**{}/{}*{}/{}** loss = {} sdf={} grad={} eik={}'.format(\
                        self.expdir, self.timestamp, epoch, self.nepochs, \
                        loss.item(), loss_output['sdf_loss'].item(), loss_output['sdf_loss'].item(), \
                        loss_output['grad_loss'].item(), loss_output['eikonal_loss'].item()))
                    else:
                        psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                            ground_truth['rgb'].cuda().reshape(-1,3))

                        print('**{}/{}*{}/{}** loss = {} rgb={} psnr={} laplace_tan={} eik={} deform={} deform_inv={} points_cir_loss={}'.format( \
                        self.expdir, self.timestamp, epoch, self.nepochs, \
                        loss.item(), loss_output['rgb_loss'].item(), psnr.item(), \
                        loss_output['laplace_tan_loss'].item(), \
                        loss_output['eikonal_loss'].item(), \
                        loss_output['deform_loss'].item(), loss_output['deform_inv_loss'].item(), \
                        loss_output['points_cir_loss'].item()))

                        writer.add_scalar('loss', loss.item(), epoch)
                        writer.add_scalar('rgb_loss', loss_output['rgb_loss'].item(), epoch)
                        writer.add_scalar('psnr', psnr.item(), epoch)
                        writer.add_scalar('eik_loss', loss_output['eikonal_loss'].item(), epoch)
                        writer.add_scalar('deform_loss', loss_output['deform_loss'].item(), epoch)
                        writer.add_scalar('deform_inv_loss', loss_output['deform_inv_loss'].item(), epoch)
                        writer.add_scalar('points_cir_loss', loss_output['points_cir_loss'].item(), epoch)
                        writer.add_scalar('laplace_tan_loss', loss_output['laplace_tan_loss'].item(), epoch)

                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)
        writer.close()


    def get_plot_data(self, model_outputs, id, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
        plot_data = {
            'id': id,
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }
        if 'z_vals' in model_outputs:
            plot_data['z_vals'] = model_outputs['z_vals']
        if 'z_samples_eik' in model_outputs:
            plot_data['z_samples_eik'] = model_outputs['z_samples_eik']
        if 'cam_c2w' in model_outputs:
            plot_data['cam_c2w'] = model_outputs['cam_c2w']
        if 'weights' in model_outputs:
            plot_data['weights'] = model_outputs['weights']

        return plot_data
