import math
from dataclasses import dataclass, field

import torch
from kiui.op import safe_normalize
from pykdtree.kdtree import KDTree

from utils.cutils import grid_subsampling
from utils.gaussian_splatting_batch import project_points, shelter_points
from utils.subsample import create_sampler, fps_sample


def points_centroid(xyz):
    """
    :param xyz: [B, N, 3]
    :return: [B, 3]
    """
    return torch.mean(xyz, dim=1)


def points_scaler(xyz, scale=1.):
    """
    :param xyz: [B, N, 3]
    :param scale: float, scale factor, by default 1.0, which means scale points xyz into [0, 1]
    :return: [B, N, 3]]
    """
    mi, ma = xyz.min(dim=1, keepdim=True)[0], xyz.max(dim=1, keepdim=True)[0]
    xyz = (xyz - mi) / (ma - mi + 1e-12)
    return xyz * scale


def look_at(campos, target, opengl=True, device='cuda'):
    """construct pose rotation matrix by look-at. a pytorch implementation @kiui

    Args:
        campos (torch.Tensor): camera position, float [3]
        target (torch.Tensor): target position to look at, float [3]
        opengl (bool, optional): whether to use opengl camera convention. by default True.
        device (str, optional): device. by default torch.device('cuda')

    Returns:
        torch.Tensor: the camera pose rotation matrix, float [3, 3], normalized.
    """
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector))
        up_vector = safe_normalize(torch.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
        up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1))
    R = torch.stack([right_vector, up_vector, forward_vector], dim=1)
    return R


class OrbitCamera:
    """ An orbital camera class. a custom pytorch implementation @kiui
    """

    def __init__(self, camid: int, width: int, height: int, campos: tuple, target: tuple = None,
                 fovy: float = 60, cam_index=-1, target_index=-1, device='cuda'):
        """init function

        Args:
            camid (int): camera id.
            width (int): view width of camera field size.
            height (int): view height of camera field size.
            campos (tuple): camera position.
            target (tuple, optional): look at target position. by default (0, 0, 0).
            fovy (float, optional): camera field of view in degree along y-axis. by default 60.
            cam_index (float, optional): the camera index in point cloud, for visualization. by default -1.
            target_index (float, optional): the target index in point cloud, for visualization. by default -1.
            device (str, optional): device. by default 'cuda'
        """
        self.camid = torch.tensor(camid, device=device)
        self.W = width
        self.H = height
        self.campos = torch.tensor(campos, dtype=torch.float32, device=device)
        self.fovy = torch.deg2rad(torch.tensor([fovy], dtype=torch.float32, device=device))[0]
        if target is None:
            target = (0, 0, 0)
        self.target = torch.tensor(target, dtype=torch.float32, device=device)  # look at this point
        self.cam_index = cam_index
        self.target_index = target_index
        self.device = device

    @property
    def fovx(self):
        return 2 * torch.arctan(torch.tan(self.fovy / 2) * self.W / self.H)

    @property
    def pose(self):
        pose = torch.eye(4, dtype=torch.float32, device=self.device)
        pose[:3, :3] = look_at(self.campos, self.target, device=self.device)
        pose[:3, 3] = self.campos
        return pose

    @property
    def intrinsics(self):
        focal = self.H / (2 * torch.tan(self.fovy / 2))
        return torch.tensor([focal, focal, self.W // 2, self.H // 2], dtype=torch.float32, device=self.device)


@dataclass
class CameraOptions(dict):
    # virtual camera numbers
    n_cameras: int = 4
    # camera field of view in degree along y-axis.
    cam_fovy: float = 60.0
    # camera field size, [width, height]
    cam_field_size: list = field(default_factory=list)
    # use shelter points
    use_shelter: bool = False
    # camera sampler, ['random', 'fps', ...]
    cam_sampler: str = 'fps'
    # generate camera method, ['centroid', 'farthest']
    cam_gen_method: str = 'centroid'

    @classmethod
    def default(cls, n_cameras=4):
        return CameraOptions(
            n_cameras=n_cameras,
            cam_fovy=120.0,
            cam_field_size=[512, 512],
            use_shelter=False,
            cam_sampler='fps',
            cam_gen_method='centroid'
        )

    def __str__(self):
        return f'''CameraOptions(
            n_cameras={self.n_cameras},
            cam_fovy={self.cam_fovy},
            cam_field_size={self.cam_field_size},
            use_shelter={self.use_shelter},
            cam_sampler={self.cam_sampler},
            cam_gen_method={self.cam_gen_method})'''


class CameraPoints(object):
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

    def __set_attr__(self, key, value):
        assert not self.__is_attr_exists__(key)
        self.__dict__[key] = value

    def __update_attr__(self, key, value):
        self.__dict__[key] = value

    def __get_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return None
        return self.__dict__[key]

    def __del_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return
        self.__dict__.pop(key)

    def __is_attr_exists__(self, key):
        return key in self.__dict__.keys()

    def keys(self):
        return self.__dict__.keys()

    def to_cuda(self, non_blocking=True):
        keys = self.keys()
        for key in keys:
            item = self.__get_attr__(key)
            if isinstance(item, torch.Tensor):
                item = item.cuda(non_blocking=non_blocking)
            if isinstance(item, list):
                for i in range(len(item)):
                    if isinstance(item[i], torch.Tensor):
                        item[i] = item[i].cuda(non_blocking=non_blocking)
            self.__update_attr__(key, item)
        self.device = 'cuda'

    @property
    def layer_idx(self):
        return self.__get_attr__('layer_idx')

    @property
    def idx_ds(self):
        return self.__get_attr__('idx_ds')

    @property
    def idx_us(self):
        return self.__get_attr__('idx_us')

    @property
    def idx_group(self):
        return self.__get_attr__('idx_group')

    @property
    def pts_list(self):
        return self.__get_attr__('pts_list')

    @property
    def p(self):
        return self.__get_attr__('p')

    @property
    def f(self):
        return self.__get_attr__('f')

    @property
    def y(self):
        return self.__get_attr__('y')

    @property
    def f_cam(self):
        return self.__get_attr__('f_cam')

    @property
    def cameras(self):
        cameras = self.__get_attr__('orbit_cameras')
        return cameras

    @property
    def uv(self):
        item = self.__get_attr__('uv')
        return item

    @property
    def depths(self):
        item = self.__get_attr__('depths')
        return item

    @property
    def visible(self):
        item = self.__get_attr__('visible')
        return item

    @property
    def cam_intr(self):
        return self.__get_attr__('cam_intr')

    @property
    def cam_extr(self):
        return self.__get_attr__('cam_extr')


def generate_cameras(xyz, opt: CameraOptions):
    """
    :param xyz: [N, 3]
    :return: n_cameras*2 cameras
    """
    gen_method = opt.cam_gen_method
    if gen_method == 'centroid':
        return generate_cameras_by_centroid(xyz, opt)
    elif gen_method == 'farthest':
        return generate_cameras_by_farthest(xyz, opt)
    else:
        raise NotImplementedError


@torch.no_grad()
def generate_cameras_by_centroid(xyz, opt):
    """
    :param xyz: [N, 3]
    :return: n_cameras*2 cameras
    """
    n_cameras = opt.n_cameras
    cam_fovy = opt.cam_fovy
    cam_width, cam_height = opt.cam_field_size
    cam_sampler = create_sampler(opt.cam_sampler)

    centroid = points_centroid(xyz.unsqueeze(0)).squeeze(0)
    xyz_sampled, _ = cam_sampler(xyz=xyz.unsqueeze(0), n_samples=n_cameras)
    xyz_sampled = xyz_sampled.squeeze(0)
    cam_intr_all = []
    cam_extr_all = []
    cameras_all = []
    for j in range(n_cameras):
        cx, cy, cz = centroid
        x, y, z = xyz_sampled[j]
        outside_c = OrbitCamera(
            camid=2 * j + 1,
            width=cam_width,
            height=cam_height,
            campos=(x, y, z),
            target=(cx, cy, cz),
            fovy=cam_fovy,
            device=xyz.device,
        )
        cam_intr = outside_c.intrinsics
        cam_extr = outside_c.pose
        cam_intr_all.append(cam_intr)
        cam_extr_all.append(cam_extr)

        inside_c = OrbitCamera(
            camid=2 * j + 2,
            width=cam_width,
            height=cam_height,
            campos=(cx, cy, cz),
            target=(x, y, z),
            fovy=cam_fovy,
            device=xyz.device,
        )
        cam_intr = inside_c.intrinsics
        cam_extr = inside_c.pose
        cam_intr_all.append(cam_intr)
        cam_extr_all.append(cam_extr)
        cameras_all.append(outside_c)
        cameras_all.append(inside_c)
    cam_intr = torch.stack(cam_intr_all, dim=0)
    cam_extr = torch.stack(cam_extr_all, dim=0)
    return cameras_all, cam_intr, cam_extr


@torch.no_grad()
def generate_cameras_by_farthest(xyz, opt):
    """
    :param xyz: [N, 3]
    :return: n_cameras*2 cameras
    """
    n_cameras = opt.n_cameras
    cam_fovy = opt.cam_fovy
    cam_width, cam_height = opt.cam_field_size
    cam_sampler = create_sampler(opt.cam_sampler)
    assert opt.cam_sampler == 'fps'

    xyz_sampled, idx = cam_sampler(xyz=xyz.unsqueeze(0), n_samples=n_cameras * 2 + 1)
    xyz_sampled = xyz_sampled.squeeze(0)
    cam_intr_all = []
    cam_extr_all = []
    cameras_all = []
    for j in range(1, n_cameras * 2 + 1):
        cx, cy, cz = xyz_sampled[j - 1]
        x, y, z = xyz_sampled[j]
        c = OrbitCamera(
            camid=j,
            width=cam_width,
            height=cam_height,
            campos=(x, y, z),
            target=(cx, cy, cz),
            fovy=cam_fovy,
            cam_index=idx[0][j],
            target_index=idx[0][j-1],
            device=xyz.device,
        )
        cam_intr = c.intrinsics
        cam_extr = c.pose
        cam_intr_all.append(cam_intr)
        cam_extr_all.append(cam_extr)
        cameras_all.append(c)
    cam_intr = torch.stack(cam_intr_all, dim=0)
    cam_extr = torch.stack(cam_extr_all, dim=0)
    return cameras_all, cam_intr, cam_extr


class CameraHelper:
    def __init__(self,
                 opt: CameraOptions = None,
                 batch_size: int = 8,
                 device: str = 'cuda',
                 **kwargs):
        if opt is None:
            opt = CameraOptions.default()
        self.opt = opt
        self.batch_size = batch_size
        self.device = device
        self.cam_points = CameraPoints(batch_size=self.batch_size, device=self.device)

    def init_points(self):
        self.cam_points = CameraPoints(batch_size=self.batch_size, device=self.device)

    def to(self, device):
        self.device = device
        return self

    @torch.no_grad()
    def projects(self, xyz, scale=1., cam_batch=128):
        """
        :param xyz: [N, 3]
        :param scale: xyz scale factor
        :param cam_batch: projection batch size of camera
        :return: [N, 3]
        """
        assert len(xyz.shape) == 2
        n_cameras = self.opt.n_cameras
        if n_cameras * 2 < cam_batch:
            cam_batch = n_cameras * 2
        cam_width, cam_height = self.opt.cam_field_size
        use_shelter = self.opt.use_shelter
        if scale > 0:
            # recommend to use scaler
            xyz_scaled = points_scaler(xyz.unsqueeze(0), scale=scale).squeeze(0)
        else:
            xyz_scaled = xyz
        cam_all, cam_intr, cam_extr = generate_cameras(xyz_scaled, self.opt)
        self.cam_points.__update_attr__('orbit_cameras', cam_all)

        extent = 1.3
        uv_all, depths_all, visible_all = [], [], []
        for j in range(n_cameras * 2 // cam_batch):
            uv, depths = project_points(
                xyz_scaled,
                cam_intr[cam_batch*j:cam_batch*j+cam_batch],
                cam_extr[cam_batch*j:cam_batch*j+cam_batch],
                cam_width,
                cam_height,
                nearest=0,
                extent=extent,
            )
            uv = uv.permute(1, 2, 0)
            depths = depths.permute(1, 2, 0)
            if use_shelter:
                mW, mH = int(extent * cam_width) + 1, int(extent * cam_height) + 1
                bucket_size = (mH + 1) * (mW + 1)
                bucket = torch.zeros([bucket_size * n_cameras * 2])
                offset = torch.arange(0, cam_batch, 1, device=xyz.device)
                uv, depths, visible = shelter_points(uv, depths, mW, mH, offset=offset, bucket=bucket)
            else:
                visible = (depths > 0).int()

            uv_all.append(uv)
            depths_all.append(depths)
            visible_all.append(visible)
        uv = torch.cat(uv_all, dim=-1)
        depths = torch.cat(depths_all, dim=-1)
        visible = torch.cat(visible_all, dim=-1)
        self.cam_points.__update_attr__('uv', uv)
        self.cam_points.__update_attr__('depths', depths)
        self.cam_points.__update_attr__('visible', visible)
        self.cam_points.__update_attr__('f_cam', visible.squeeze().float())


def calc_distance_scaler(full_p):
    # estimating a distance in Euclidean space as the scaler by random fps
    ps, _ = fps_sample(full_p.unsqueeze(0), 2, random_start_point=True)
    ps = ps.squeeze(0)
    p0, p1 = ps[0], ps[1]
    scaler = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
    return scaler


def make_cam_points(cam_points, ks, grid_size=None, n_samples=None, up_sample=True, alpha=0.) -> CameraPoints:
    assert (grid_size is not None and n_samples is not None) is False
    assert (grid_size is None and n_samples is None) is False
    n_layers = len(ks)
    full_p = cam_points.p
    full_visible = cam_points.visible.squeeze().byte()
    scaler = calc_distance_scaler(full_p)

    full_p = full_p.contiguous()
    full_visible = full_visible.contiguous()
    visible = full_visible
    p = full_p

    idx_ds = []
    idx_us = []
    idx_group = []
    idx_gs_group = []
    for i in range(n_layers):
        # down sample by grid sample or fps
        if i > 0:
            if grid_size is not None:
                gsize = grid_size[i-1]
                if p.is_cuda:
                    ds_idx = grid_subsampling(p.detach().cpu(), gsize)
                else:
                    ds_idx = grid_subsampling(p, gsize)
            else:
                _, ds_idx = fps_sample(p.unsqueeze(0), n_samples[i-1])
                ds_idx = ds_idx.squeeze(0)
            p = p[ds_idx]
            visible = visible[ds_idx]
            idx_ds.append(ds_idx)

        # knn group by kdtree
        k = ks[i]
        kdt = KDTree(p.numpy(), visible.numpy())
        _, idx = kdt.query(p.numpy(), visible.numpy(), k=k, alpha=alpha, scaler=scaler)
        idx_group.append(torch.from_numpy(idx).long())

        # up sample by nn interpolation
        if i > 0 and up_sample:
            _, us_idx = kdt.query(full_p.numpy(), full_visible.numpy(), k=1, alpha=alpha, scaler=scaler)
            idx_us.append(torch.from_numpy(us_idx).long())

    cam_points.__update_attr__('idx_ds', idx_ds)
    cam_points.__update_attr__('idx_us', idx_us)
    cam_points.__update_attr__('idx_group', idx_group)
    cam_points.__update_attr__('idx_gs_group', idx_gs_group)
    return cam_points


def merge_cam_points(cam_points_list, up_sample=True) -> CameraPoints:
    assert len(cam_points_list) > 0
    new_cam_points = CameraPoints(batch_size=len(cam_points_list),
                                  device=cam_points_list[0].device)

    p_all = []
    f_cam_all = []
    f_all = []
    rgb_all = []
    y_all = []
    idx_ds_all = []
    idx_us_all = []
    idx_group_all = []
    pts_all = []
    n_layers = len(cam_points_list[0].idx_group)
    pts_per_layer = [0] * n_layers
    for i in range(len(cam_points_list)):
        cam_points = cam_points_list[i]
        p_all.append(cam_points.p)
        f_cam_all.append(cam_points.f_cam)
        f_all.append(cam_points.f)
        rgb = cam_points.__get_attr__('rgb')
        if rgb is not None:
            rgb_all.append(rgb)
        y_all.append(cam_points.y)

        idx_ds = cam_points.idx_ds
        idx_us = cam_points.idx_us
        idx_group = cam_points.idx_group
        pts = []
        for layer_idx in range(n_layers):
            if layer_idx < len(idx_ds):
                idx_ds[layer_idx].add_(pts_per_layer[layer_idx])
                if up_sample:
                    idx_us[layer_idx].add_(pts_per_layer[layer_idx + 1])
            idx_group[layer_idx].add_(pts_per_layer[layer_idx])
            pts.append(idx_group[layer_idx].shape[0])
        idx_ds_all.append(idx_ds)
        idx_us_all.append(idx_us)
        idx_group_all.append(idx_group)
        pts_all.append(pts)
        pts_per_layer = [pt + idx.shape[0] for (pt, idx) in zip(pts_per_layer, idx_group)]

    p = torch.cat(p_all, dim=0)
    new_cam_points.__update_attr__('p', p)

    f_cam = torch.cat(f_cam_all, dim=0)
    new_cam_points.__update_attr__('f_cam', f_cam)

    f = torch.cat(f_all, dim=0)
    new_cam_points.__update_attr__('f', f)

    if len(rgb_all) > 0:
        rgb = torch.cat(rgb_all, dim=0)
        new_cam_points.__update_attr__('rgb', rgb)

    y = torch.cat(y_all, dim=0)
    new_cam_points.__update_attr__('y', y)

    # layer_idx is [1, 2, 3] when CamPoint layers = 4
    idx_ds = [torch.cat(idx, dim=0) for idx in zip(*idx_ds_all)]
    new_cam_points.__update_attr__('idx_ds', idx_ds)

    # layer_idx is [2, 1, 0] when CamPoint layers = 4
    idx_us = [torch.cat(idx, dim=0) for idx in zip(*idx_us_all)]
    new_cam_points.__update_attr__('idx_us', idx_us)

    # layer_idx is [0, 1, 2, 3] when CamPoint layers = 4
    idx_group = [torch.cat(idx, dim=0) for idx in zip(*idx_group_all)]
    new_cam_points.__update_attr__('idx_group', idx_group)

    # batch_size * layer_idx is [0, 1, 2, 3] when CamPoint layers = 4
    pts_list = torch.tensor(pts_all, dtype=torch.int64)
    pts_list = pts_list.view(-1, n_layers).transpose(0, 1).contiguous()
    new_cam_points.__update_attr__('pts_list', pts_list)
    return new_cam_points
