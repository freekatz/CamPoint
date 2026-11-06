import torch
from einops import repeat, rearrange
from numba import njit


def project_points(
        xyz: torch.Tensor,
        intr: torch.Tensor,
        extr: torch.Tensor,
        W: int,
        H: int,
        nearest: float = 0.2,
        extent: float = 1.3,
):
    """Project 3D points to the screen.
        a pytorch implementation support batch @msplat

    Args:
        xyz (torch.Tensor): points, [M, N, 3] or [N, 3], M is camera numbers
        intr (torch.Tensor): camera intrinsics, [M, 4]
        extr (torch.Tensor): OpenGL camera-to-world pose, [M, 4, 4]
        W (int): view width of camera field size.
        H (int): view height of camera field size.
        nearest (float, optional): nearest threshold for frustum culling, by default 0.2
        extent (float, optional): extent threshold for frustum culling, by default 1.3

    Returns
        uv (torch.Tensor): 2D positions for each point in the image, [M, N, 2]
        depth (torch.Tensor): depth for each point, [M, N, 1]
    """
    assert len(intr.shape) == 2
    M = intr.shape[0]
    device = xyz.device

    K = torch.eye(3, device=device)
    K = repeat(K, 'n d -> m n d', m=M)
    K = K.clone()
    K[:, 0, 0] = intr[:, 0]
    K[:, 1, 1] = intr[:, 1]
    K[:, 0, 2] = intr[:, 2]
    K[:, 1, 2] = intr[:, 3]
    R = extr[:, :3, :3]
    t = extr[:, :3, -1].unsqueeze(dim=-1)

    xyz_t = xyz.permute(0, 2, 1) if len(xyz.shape) == 3 else xyz.permute(1, 0)
    pt_cam = torch.matmul(R.transpose(-1, -2), xyz_t - t)
    # Convert OpenGL camera axes to x-right, y-down, z-forward.
    pt_cam[:, 1:3, :] *= -1
    # Apply camera intrinsic matrix
    p_proj = torch.matmul(K[:, :], pt_cam)

    depths = repeat(p_proj[:, 2], 'b m -> b d m', d=2)
    uv = p_proj[:, :2] / depths - 0.5
    uv = uv.permute(0, 2, 1)

    depths = torch.nan_to_num(depths[:, 0]).squeeze(-1)
    extent_mask_x = torch.logical_or(uv[:, :, 0] < (1 - extent) * W * 0.5,
                                     uv[:, :, 0] > (1 + extent) * W * 0.5)
    extent_mask_y = torch.logical_or(uv[:, :, 1] < (1 - extent) * H * 0.5,
                                     uv[:, :, 1] > (1 + extent) * H * 0.5)
    extent_mask = torch.logical_or(extent_mask_x, extent_mask_y)
    mask = torch.logical_or(extent_mask, depths <= 0)
    if nearest > 0:
        near_mask = depths <= nearest
        mask = torch.logical_or(near_mask, mask)

    uv_masked = uv.clone()
    depths_masked = depths.clone()
    uv_masked[:, :, 0][mask] = 0
    uv_masked[:, :, 1][mask] = 0
    depths_masked[mask] = 0

    return uv_masked, depths_masked.unsqueeze(-1)


@njit(fastmath=True)
def _shelter_points(
        uv,
        xy,
        depths,
        bucket,
        mH, mW,
        N, M,
):
    """
    :param uv: [N, 2, M]
    :param xy: [N, 2, M]
    :param depths: [N, 1, M]
    :param bucket: [M * mH * mW, 1]
    :return: None, modify uv and depths inplace
    """
    for i in range(N):
        for j in range(M):
            x = xy[i, 0, j]
            y = xy[i, 1, j]
            d = depths[i, 0, j]
            idx = (mH + 1) * x + y
            _d = bucket[idx]

            if _d == 0:
                bucket[idx] = d
                continue
            if d < _d and d != 0:
                bucket[idx] = d
    for i in range(N):
        for j in range(M):
            x = xy[i, 0, j]
            y = xy[i, 1, j]
            d = depths[i, 0, j]
            idx = (mH + 1) * x + y
            _d = bucket[idx]
            if d > _d and d != 0:
                uv[i, 0, j] = 0
                uv[i, 1, j] = 0
                depths[i, 0, j] = 0


def shelter_points(uv, depths, mW, mH, offset, bucket):
    """
    :param uv: [N, 2, M]
    :param depths: [N, 1, M]
    :param bucket: [M * mH * mW, 1]
    :return: [N, 2, M], [N, 1, M], [N, 1, M]
    """
    N = uv.shape[0]
    M = uv.shape[-1]

    _uv = uv.add(offset)
    delta = torch.round((_uv % 1) * 1e5) / 1e5
    xy = _uv - delta
    xy = xy.squeeze() - xy.min()
    xy = xy.long()
    bucket[:] = 0

    _shelter_points(
        uv.cpu().numpy(), xy.cpu().numpy(), depths.cpu().numpy(), bucket.cpu().numpy(),
        mW, mH, N, M
    )
    visible = depths != 0
    return uv, depths, visible


def shelter_points2(uv_origin, depths_origin, mW, mH, offset, bucket):
    """
    :param uv: [N, 2, M]
    :param depths: [N, 1, M]
    :param bucket: [M * mH * mW, 1]
    :return: [N, 2, M], [N, 1, M], [N, 1, M]
    """
    N = uv_origin.shape[0]
    M = uv_origin.shape[-1]

    uv = uv_origin.clone().add(offset)

    uv_origin = rearrange(uv_origin, 'n c m -> (n m) c')
    uv = rearrange(uv, 'n c m -> (n m) c')
    depths_origin = rearrange(depths_origin.clone(), 'n c m -> (n m) c')
    depths_origin = depths_origin.squeeze()

    shelter_idx = torch.nonzero(depths_origin != 0)
    uv = uv[shelter_idx].squeeze()
    depths = depths_origin[shelter_idx].squeeze()

    delta = torch.round((uv % 1) * 1e5) / 1e5
    xy = uv - delta
    xy = xy.squeeze() - xy.min()
    xy = xy.long()

    bucket[:] = 0
    sorted_idx = torch.argsort(depths, dim=0, descending=True)
    sorted_xy = xy[sorted_idx, :]
    sorded_depths = depths[sorted_idx]
    sorted_xy = sorted_xy.permute(1, 0)
    idx = (mH + 1) * sorted_xy[0, :] + sorted_xy[1, :]
    torch.index_put_(bucket, list([idx]), sorded_depths)

    xy_inv = xy.permute(1, 0)
    idx_inv = (mH + 1) * xy_inv[0, :] + xy_inv[1, :]
    new_depths = bucket[idx_inv]

    mask = torch.logical_or(new_depths != depths, depths == 0)
    shelter_idx = torch.masked_select(shelter_idx.squeeze(), mask)
    uv_origin[shelter_idx, :] = 0
    depths_origin[shelter_idx] = 0
    visible = depths_origin != 0

    new_uv = rearrange(uv_origin, '(n m) c -> n c m', n=N, m=M)
    new_depths = rearrange(depths_origin, '(n c m) -> n c m', n=N, m=M, c=1)
    visible = rearrange(visible, '(n c m) -> n c m', n=N, m=M, c=1)
    return new_uv, new_depths, visible


if __name__ == '__main__':
    xyz = torch.randn(10, 3)
    intr = torch.randn(2, 4)
    extr = torch.randn(2, 4, 4)
    uv, depths = project_points(xyz, intr, extr, W=512, H=512)
    print(uv.shape, depths.shape)

    extent = 1.3
    cam_width = cam_height = 512
    n_cameras = 1
    cam_batch = n_cameras * 2

    uv = torch.tensor([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[2, 2], [2, 2]], [[2, 2], [2, 2]]])
    depths = torch.tensor([[[0.2], [0.2]], [[0.3], [0.3]], [[0], [0]], [[0], [0.1]]])
    uv = uv.permute(0, 2, 1)
    depths = depths.permute(0, 2, 1)
    uv = uv.contiguous()
    depths = depths.contiguous()

    print(uv)
    print(depths)

    mW, mH = int((1 + extent) * cam_width) + 1, int((1 + extent) * cam_height) + 1
    bucket_size = (mH + 1) * (mW + 1)
    bucket = torch.zeros([bucket_size * n_cameras * 2])
    offset = torch.arange(0, cam_batch, 1, device=uv.device)

    uv, depths, visible = shelter_points(uv, depths, mH, mW, offset, bucket)
    print(uv)
    print(depths)
    print(visible)
