# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import time
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn
import yaml

from . import models
from .bfm import BFMModel
from .utils.io import _load
from .utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from utils.tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz
)

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)

"""
    TDDFA_V2をpip installして利用できるようにしている
    https://github.com/cleardusk/3DDFA_V2

    変更方針
    * 以下前提でコードの変更をしている
        * モデルの追加変更はないこと
        * 使用するモデルをmobilenetから変えないこと
    * demo関係と内包されていたfaceboxesはdemo dirに移動
    * utilsとbfmはそのまま移動
        * 両方とも最小限のパス通ししかしていないので、そのままでは動かないものが存在する点注意

    is_defaultを追加しているのは、使用したいときに、configファイルへのパスを意識しなくてもいいようにするため
    もしモデルの変更追加を行うならば、**kvsで設定値を受け取るよりも、configsに設定ファイルが置くようにして、
    `config_path="configs/mb1_120x120.yml"` だけを引数で渡して、__init__内でyml load->設定値として使う形式にした方が良い。
    ->ただし変更範囲は少し広くなるし、kvs.get()のデフォルト値の見直しも必要
"""


default_cfg = yaml.load(open(make_abs_path("configs/mb1_120x120.yml")), Loader=yaml.SafeLoader)

class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, *, is_default=False, **kvs):
        """
        Args:
            is_default (bool, optional): 使用するモデルをデフォルトのmobilenetv1ベースのものにするかどうか. Defaults to False.
            **kvs: 辞書型で設定値を渡す. configファイルをyml loadしたものが入る想定
        """

        torch.set_grad_enabled(False)

        if is_default:
            kvs = default_cfg

        # load BFM
        self.bfm = BFMModel(
            # make_abs_pathの掛け方を変更している場所がある
            # tddfa/configs, tddfa/weightsの中にモデルなどをおくことを前提とするため
            # kvs.get('bfm_fp') などで相対パスを取得してしまった場合、上記のディレクトリまで到達できなくはないけど大変なため、その対策
            # もしtddfaの外に設定値、モデルを置く場合は、元のコードに戻して、configファイルで指定する形にすると良い
            bfm_fp=make_abs_path(kvs.get('bfm_fp', 'configs/bfm_noneck_v3.pkl')),
            shape_dim=kvs.get('shape_dim', 40),
            exp_dim=kvs.get('exp_dim', 10)
        )
        self.tri = self.bfm.tri

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = make_abs_path(kvs.get(
            'param_mean_std_fp', f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),
            widen_factor=kvs.get('widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        model = load_model(model, make_abs_path(kvs.get('checkpoint_fp')))

        if self.gpu_mode:
            cudnn.benchmark = True
            model = model.cuda(device=self.gpu_id)

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def __call__(self, img_ori, objs, **kvs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: the list of box or landmarks
        :param kvs: options
        :return: param list and roi_box list
        """
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            inp = self.transform(img).unsqueeze(0)

            if self.gpu_mode:
                inp = inp.cuda(device=self.gpu_id)

            if kvs.get('timer_flag', False):
                end = time.time()
                param = self.model(inp)
                elapse = f'Inference: {(time.time() - end) * 1000:.1f}ms'
                print(elapse)
            else:
                param = self.model(inp)

            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            # print('output', param)
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            if dense_flag:
                R, offset, alpha_shp, alpha_exp = _parse_param(param)
                pts3d = R @ (self.bfm.u + self.bfm.w_shp @ alpha_shp + self.bfm.w_exp @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)
            else:
                R, offset, alpha_shp, alpha_exp = _parse_param(param)
                pts3d = R @ (self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)

            ver_lst.append(pts3d)

        return ver_lst
