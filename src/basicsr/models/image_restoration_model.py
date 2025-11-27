# from Restormer:
# https://github.com/swz30/Restormer/blob/7a36b56f0d3704364f5c77352c11ac817223fe2e/basicsr/models/image_restoration_model.py#L51
import importlib

import clip
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

from tqdm import tqdm

# from basicsr.models.hook_visualize import HookTool, featuremap_2_heatmap
from basicsr.utils import get_root_logger, imwrite, tensor2img

from basicsr import define_network
from basicsr.models.base_model import BaseModel, BaseModel_Restormer

loss_module = importlib.import_module('basicsr.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class ImageCleanModel_IterSup_Vec(BaseModel_Restormer):

    def load_clip_model(self, device='cpu'):
        from clip import clip
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.float()

        for para in model.parameters():
            para.requires_grad = False
        return model, preprocess
    
    def __init__(self, opt):
        super(ImageCleanModel_IterSup_Vec, self).__init__(opt)

        clip_model, _ = self.load_clip_model()
        net_opt = deepcopy(opt['network_g'])
        net_opt['model_clip'] =  clip_model
        self.net_opt = deepcopy(net_opt)

        # define network
        # self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = define_network(net_opt)
        self.net_g = self.model_to_device(self.net_g)

        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            # self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            self.net_g_ema = define_network(self.net_opt).to(self.device)       
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)

        if train_opt.get('clip_opt'):                                       
            clip_type = train_opt['clip_opt'].pop('type')
            cri_clip_cls = getattr(loss_module, clip_type)
            self.clip_type = clip_type
            self.cri_clip = cri_clip_cls(**train_opt['clip_opt']).to(self.device)
        else:
            self.cri_clip = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)


    def tokenize_text(self, texts):

        tokenized_texts = torch.cat([clip.tokenize(text) for text in texts]).cuda()
        return tokenized_texts


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'text' in data:
            text = data['text']
            self.text = text
        self.tokenized_texts = self.tokenize_text(text)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds, out_li = self.net_g(self.lq, self.tokenized_texts)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        for idx, out in enumerate(out_li):
            # pixel loss
            l_pix = 0
            l_pix = self.cri_pix(out, self.gt)
            l_total += l_pix
            loss_dict[f'l_pix_iter{idx}'] = l_pix

        l_total.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                n = len(self.lq)
                outs = []
                i = 0
                while i<n :
                    pred, out_li =  self.net_g_ema(img[i].unsqueeze(0), self.tokenized_texts)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    i = i + 1
                self.output = torch.cat(outs, dim=0)

        else:
            self.net_g.eval()
            with torch.no_grad():
                n = len(self.lq)
                outs = []
                i = 0
                while i < n :
                    pred, out_li = self.net_g(img[i].unsqueeze(0), self.tokenized_texts)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    i = i + 1
                self.output = torch.cat(outs, dim=0)

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')
            self._initialize_best_metric_results(dataset_name)

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['img_path'][0]))[0]

            self.feed_data(val_data)

            if self.opt['val'].get('grids', False):
                self.grids()

            test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name,f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                # imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
                ###### 
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
                
            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _initialize_best_metric_results(self, dataset_name):
            """Initialize the best metric results dict for recording the best metric value and iteration."""
            if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
                return
            elif not hasattr(self, 'best_metric_results'):
                self.best_metric_results = dict()

            # add a dataset record
            record = dict()
            for metric, content in self.opt['val']['metrics'].items():
                better = content.get('better', 'higher')
                init_val = float('-inf') if better == 'higher' else float('inf')
                record[metric] = dict(better=better, val=init_val, iter=-1)
            self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
                
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def grids(self):
            b, c, h, w = self.gt.size()
            self.original_size = (b, c, h, w)

            assert b == 1
            if 'crop_size_h' in self.opt['val']:
                crop_size_h = self.opt['val']['crop_size_h']
            else:
                crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

            if 'crop_size_w' in self.opt['val']:
                crop_size_w = self.opt['val'].get('crop_size_w')
            else:
                crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


            # crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
            crop_size_h, crop_size_w = crop_size_h , crop_size_w
            #adaptive step_i, step_j
            num_row = (h - 1) // crop_size_h + 1
            num_col = (w - 1) // crop_size_w + 1

            import math
            step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
            step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

            scale = self.opt.get('scale', 1)

            step_i = step_i//scale*scale
            step_j = step_j//scale*scale

            parts = []
            idxes = []

            i = 0  # 0~h-1
            last_i = False
            while i < h and not last_i:
                j = 0
                if i + crop_size_h >= h:
                    i = h - crop_size_h
                    last_i = True

                last_j = False
                while j < w and not last_j:
                    if j + crop_size_w >= w:
                        j = w - crop_size_w
                        last_j = True
                    parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                    idxes.append({'i': i, 'j': j})
                    j = j + step_j
                i = i + step_i

            self.origin_lq = self.lq
            self.lq = torch.cat(parts, dim=0)
            self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        # crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        crop_size_h, crop_size_w = crop_size_h, crop_size_w


        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.output[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq