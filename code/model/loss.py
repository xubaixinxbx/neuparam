import torch
from torch import nn
import utils.general as utils
import torch.nn.functional as F

class VolSDFLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eikonal_weight = kwargs['eikonal_weight']
        
        if 'latent_code_loss' in kwargs:
            self.latent_code_loss = utils.get_class(kwargs['latent_code_loss'])(reduction='mean')
        else:
            self.latent_code_loss = None
        self.shape_code_weight = kwargs.get('shape_code_weight', 0)
        self.color_code_weight = kwargs.get('color_code_weight', 0)

        # define loss for parametric domain to overfit point cloud
        self.sdf_weight = kwargs.get('sdf_weight', 0)
        self.grad_weight = kwargs.get('grad_weight', 0)

        # define loss for parameterization given a set of multi view images
        if 'rgb_loss' in kwargs:
            self.rgb_loss = utils.get_class(kwargs['rgb_loss'])(reduction='mean')
        else:
            self.rgb_loss = None
        self.deform_weight = kwargs.get('deform_weight', 0)
        self.deform_inv_weight = kwargs.get('deform_inv_weight', 0)
        self.points_cir_weight = kwargs.get('points_cir_weight', 0)
        self.laplace_tan_weight = kwargs.get('laplace_tan_weight', 0)
        self.laplace_nor_weight = kwargs.get('laplace_nor_weight', 0)
        # shading sparsity
        self.shading_sparsity_weight = kwargs.get('shading_sparsity_weight', 0)
        
        self.shape_code_weight = kwargs.get('shape_code_weight', 0)
        self.color_code_weight = kwargs.get('color_code_weight', 0)

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_latent_code_loss(self, code):
        latent_code_loss = self.latent_code_loss(code, torch.zeros_like(code).cuda().float())
        return latent_code_loss

    def forward(self, model_outputs, ground_truth):
        #preprocess, we may need to match the shape of sdf/gradients for generated points
        if 'sdf' in model_outputs and 'sdf' in ground_truth:
            ground_truth['sdf'] = ground_truth['sdf'].squeeze(0)
            assert model_outputs['sdf'].shape[0] == ground_truth['sdf'].shape[0], print('in loss, sdf shape doesnt match')
        if 'grad_theta' in model_outputs and 'grad' in ground_truth:
            ground_truth['grad'] = ground_truth['grad'].squeeze(0)
            assert model_outputs['sdf'].shape[0] == ground_truth['sdf'].shape[0], print('in loss, sdf shape doesnt match')
        
        # overfit point cloud input in parametric domain design,  
        if 'sdf' in model_outputs:
            if 'sdf' in ground_truth:
                gt_sdf = ground_truth['sdf'].cuda()
            else:
                gt_sdf = torch.zeros_like(model_outputs['sdf']).cuda()
            pred_sdf = model_outputs['sdf']
            sdf_loss = torch.where(gt_sdf != -1, torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(pred_sdf)).abs().mean()
        else:
            sdf_loss = torch.tensor(0.0).cuda().float()
        
        # for multiview input
        if 'rgb' in ground_truth:
            rgb_gt = ground_truth['rgb'].cuda()
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        else:
            rgb_loss = torch.tensor(0.0).cuda().float()

        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
            if 'grad' in ground_truth:
                # param domain caculates grad loss
                gt_sdf = ground_truth['sdf'].cuda()
                grad_loss = (1.0 - F.cosine_similarity(model_outputs['grad_theta'], ground_truth['grad'].cuda(), dim=-1)).mean()
            else:
                grad_loss = torch.tensor(0.0).cuda().float()
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()
            grad_loss = torch.tensor(0.0).cuda().float()

        if 'domain_diff' in model_outputs:
            domain_sdf_loss = torch.abs(model_outputs['domain_diff']).mean()
        else:
            domain_sdf_loss = torch.tensor(0.0).cuda().float()

        if 'curvature_loss' in model_outputs:
            curvature_loss = model_outputs['curvature_loss']
        else:
            curvature_loss = torch.tensor(0.0).cuda().float()
        # TODO: add to total loss
        if 'code' in model_outputs:
            if 'shape_code' in model_outputs['code']:
                shape_code_loss = self.get_latent_code_loss(model_outputs['code']['shape_code'])
            else:
                shape_code_loss = torch.tensor(0.0).cuda().float()
            if 'color_code' in model_outputs['code']:
                color_code_loss = self.get_latent_code_loss(model_outputs['code']['color_code'])
            else:
                color_code_loss = torch.tensor(0.0).cuda().float()
        else:
            shape_code_loss = torch.tensor(0.0).cuda().float()
            color_code_loss = torch.tensor(0.0).cuda().float()

        if 'deform' in model_outputs:
            deform_loss = model_outputs['deform'].norm(2, dim=-1).mean()
        else:
            deform_loss = torch.tensor(0.0).cuda().float()

        if 'deform_inv' in model_outputs:
            deform_inv_loss = model_outputs['deform_inv'].norm(2, dim=-1).mean()
        else:
            deform_inv_loss = torch.tensor(0.0).cuda().float()

        if 'points_cir' in model_outputs:
            points_cir_loss = (model_outputs['points_cir']).norm(2, dim=-1).mean()
        else:
            points_cir_loss = torch.tensor(0.0).cuda().float()

        if 'shading_values' in model_outputs:
            shading_sparsity_loss = (model_outputs['shading_values']).norm(1, dim=-1).mean()
        else:
            shading_sparsity_loss = torch.tensor(0.0).cuda().float()
        
        if 'laplace_loss_tan' in model_outputs:
            # [n,3]
            laplace_tan_loss = (model_outputs['laplace_loss_tan']).norm(2, dim=-1).mean()
        else:
            laplace_tan_loss = torch.tensor(0.0).cuda().float()

        loss = rgb_loss + \
               self.sdf_weight * sdf_loss + \
               self.grad_weight * grad_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.shape_code_weight * shape_code_loss + \
               self.color_code_weight * color_code_loss + \
               self.deform_weight * deform_loss + \
               self.deform_inv_weight * deform_inv_loss + \
               self.points_cir_weight * points_cir_loss + \
               self.laplace_tan_weight * laplace_tan_loss + \
               self.shading_sparsity_weight * shading_sparsity_loss 

        output = {
            'loss': loss,
            'sdf_loss': sdf_loss,
            'grad_loss': grad_loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'shape_code_loss': shape_code_loss,
            'color_code_loss': color_code_loss,
            'deform_loss': deform_loss,
            'deform_inv_loss': deform_inv_loss,
            'points_cir_loss': points_cir_loss,
            'laplace_tan_loss': laplace_tan_loss,
            'shading_sparsity_loss': shading_sparsity_loss
        }

        return output
