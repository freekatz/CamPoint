import __init__

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from pykdtree.kdtree import KDTree

from backbone.camera import CameraOptions, CameraHelper, calc_distance_scaler, make_cam_points, merge_cam_points
from utils.cutils import grid_subsampling, grid_subsampling_test


class S3DIS(Dataset):
    classes = [
        'ceiling', 'floor', 'wall', 'beam',
        'column', 'window', 'door', 'chair',
        'table', 'bookcase', 'sofa', 'board',
        'clutter']
    num_classes = len(classes)
    class2color = {'ceiling': (23., 190., 207.),
                   'floor': (152., 223., 138.),
                   'wall': (174., 199., 232.),
                   'beam': (255., 187., 120.),
                   'column': (255., 127., 14.),
                   'window': (197., 176., 213.),
                   'door': (214., 39., 40.),
                   'table': (255., 152., 150.),
                   'chair': (188., 189., 34.),
                   'sofa': (140., 86., 75.),
                   'bookcase': (148., 103., 189.),
                   'board': (196., 156., 148.),
                   'clutter': [0, 0, 0]}
    cmap = [*class2color.values()]

    def __init__(self,
                 dataset_dir: Path,
                 area="!5",
                 loop=30,
                 train=True,
                 warmup=False,
                 voxel_max=24000,
                 k=[24, 24, 24, 24],
                 grid_size=[0.08, 0.16, 0.32],
                 alpha=0.,
                 batch_size=8,
                 cam_opts: CameraOptions = CameraOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)
        self.data_paths = list(dataset_dir.glob(f'[{area}]*.pt'))
        self.loop = loop
        self.train = train
        self.warmup = warmup
        self.voxel_max = voxel_max
        self.k = k
        self.grid_size = grid_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.cam_opts = cam_opts

        assert len(self.data_paths) > 0

        if train and warmup:
            max_n = 0
            selected_data = self.data_paths[0]
            for data in self.data_paths:
                n = torch.load(data)[0].shape[0]
                if n > max_n:
                    max_n = n
                    selected_data = data
            # use selected data with max n to warmup model
            self.data_paths = [selected_data]

        self.datas = [torch.load(path) for path in self.data_paths]

    def __len__(self):
        return len(self.data_paths) * self.loop

    @classmethod
    def get_classes(cls):
        return cls.classes

    def __getitem__(self, idx):
        if not self.train:
            return self.get_test_item(idx)

        idx //= self.loop
        xyz, col, lbl = self.datas[idx]

        angle = random.random() * 2 * math.pi
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        rotmat *= random.uniform(0.8, 1.2)
        xyz = xyz @ rotmat
        xyz += torch.empty_like(xyz).normal_(std=0.005)
        xyz -= xyz.min(dim=0)[0]

        # here grid size is assumed 0.04, so estimated downsampling ratio is ~14
        indices = grid_subsampling(xyz, 0.04, 2.5 / 14)

        xyz = xyz[indices]
        if xyz.shape[0] > self.voxel_max:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.voxel_max].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            indices = indices[condition]

        col = col[indices].float()
        rgb = col.clone()
        lbl = lbl[indices]

        if random.random() < 0.2:
            col.fill_(0.)
        else:
            if random.random() < 0.2:
                colmin = col.min(dim=0, keepdim=True)[0]
                colmax = col.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                col = (1 - alpha + alpha * scale) * col - alpha * colmin * scale
            col.mul_(1 / 250.)

        height = xyz[:, 2:]
        feature = torch.cat([col, height], dim=1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.projects(xyz)
        cam_helper.cam_points.__update_attr__('p', xyz)
        cam_helper.cam_points = make_cam_points(cam_helper.cam_points, self.k, self.grid_size,
                                                None, up_sample=True, alpha=self.alpha)
        cam_helper.cam_points.__update_attr__('f', feature)
        cam_helper.cam_points.__update_attr__('y', lbl)
        cam_helper.cam_points.__update_attr__('rgb', rgb)
        return cam_helper.cam_points

    def get_test_item(self, idx):
        pick = idx % self.loop * 5

        idx //= self.loop
        xyz, col, lbl = self.datas[idx]

        full_xyz = xyz.clone()
        full_lbl = lbl.clone()

        indices = grid_subsampling_test(xyz, 0.04, 2.5 / 14, pick=pick)
        xyz = xyz[indices]
        col = col[indices].float()
        rgb = col.clone()

        visible = np.zeros(xyz.shape[0], dtype=np.int8)
        full_visible = np.zeros(full_xyz.shape[0], dtype=np.int8)
        scaler = calc_distance_scaler(xyz)
        kdt = KDTree(xyz.numpy(), visible)
        _, full_nn = kdt.query(full_xyz.numpy(), full_visible, k=1, alpha=0., scaler=scaler)
        full_nn = torch.from_numpy(full_nn).long()

        col.mul_(1 / 250.)
        xyz -= xyz.min(dim=0)[0]
        feature = torch.cat([col, xyz[:, 2:]], dim=1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.projects(xyz)
        cam_helper.cam_points.__update_attr__('p', xyz)
        cam_helper.cam_points = make_cam_points(cam_helper.cam_points, self.k, self.grid_size,
                                                None, up_sample=True, alpha=self.alpha)
        cam_helper.cam_points.__update_attr__('f', feature)
        cam_helper.cam_points.__update_attr__('y', lbl[indices])
        cam_helper.cam_points.__update_attr__('rgb', rgb)

        return cam_helper.cam_points, full_nn, full_lbl


def s3dis_collate_fn(batch):
    cam_points_list = list(batch)
    new_cam_points = merge_cam_points(cam_points_list)
    return new_cam_points


def s3dis_test_collate_fn(batch):
    cam_points_list = [item[0] for item in batch]
    full_nn_list = [item[1] for item in batch]
    full_lbl_list = [item[2] for item in batch]

    new_cam_points = merge_cam_points(cam_points_list)
    full_nn = full_nn_list[0]
    full_lbl = full_lbl_list[0]

    return new_cam_points, full_nn, full_lbl
