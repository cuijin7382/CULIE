
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# def lut_transform(imgs, luts):
#     # img (b, 3, h, w), lut (b, c, m, m, m)
#
#     # normalize pixel values
#     imgs = (imgs - .5) * 2.
#     # reshape img to grid of shape (b, 1, h, w, 3)
#     grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)
#
#     # after gridsampling, output is of shape (b, c, 1, h, w)
#     outs = F.grid_sample(luts, grids,
#         mode='trilinear', padding_mode='border', align_corners=True)
#     # remove the extra dimension
#     outs = outs.squeeze(2)
#     # print(outs.shape)torch.Size([2, 3, 60, 60])
#     return outs
# def lut_transform(imgs, luts):
#     """
#     imgs: [B, 3, H, W] in [0,1]
#     luts: [B, R, 3, m, m, m]
#     return: [B, R, 3, H, W]
#     """
#     B, C, H, W = imgs.shape
#     R = luts.shape[1]   # rank 数量
#     m = luts.shape[3]
#     device = imgs.device
#
#     # Normalize RGB to [0, m-1]
#     grid = (imgs.permute(0, 2, 3, 1) * (m - 1)).clamp(0, m - 1 - 1e-6)  # [B, H, W, 3]
#     x, y, z = grid.unbind(-1)
#
#     x0, y0, z0 = x.floor().long(), y.floor().long(), z.floor().long()
#     x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
#     x0 = x0.clamp(0, m-1); x1 = x1.clamp(0, m-1)
#     y0 = y0.clamp(0, m-1); y1 = y1.clamp(0, m-1)
#     z0 = z0.clamp(0, m-1); z1 = z1.clamp(0, m-1)
#
#     xd, yd, zd = x - x0.float(), y - y0.float(), z - z0.float()
#
#     # 初始化输出 [B, R, 3, H, W]
#     out = torch.zeros(B, R, 3, H, W, device=device)
#
#     for b in range(B):
#         for r in range(R):
#             for c in range(3):
#                 lut = luts[b, r, c]  # [m, m, m]
#
#                 c000 = lut[x0[b], y0[b], z0[b]]
#                 c001 = lut[x0[b], y0[b], z1[b]]
#                 c010 = lut[x0[b], y1[b], z0[b]]
#                 c011 = lut[x0[b], y1[b], z1[b]]
#                 c100 = lut[x1[b], y0[b], z0[b]]
#                 c101 = lut[x1[b], y0[b], z1[b]]
#                 c110 = lut[x1[b], y1[b], z0[b]]
#                 c111 = lut[x1[b], y1[b], z1[b]]
#
#                 c00 = c000 * (1 - xd[b]) + c100 * xd[b]
#                 c01 = c001 * (1 - xd[b]) + c101 * xd[b]
#                 c10 = c010 * (1 - xd[b]) + c110 * xd[b]
#                 c11 = c011 * (1 - xd[b]) + c111 * xd[b]
#
#                 c0 = c00 * (1 - yd[b]) + c10 * yd[b]
#                 c1 = c01 * (1 - yd[b]) + c11 * yd[b]
#
#                 out[b, r, c] = c0 * (1 - zd[b]) + c1 * zd[b]
#
#     return out  # [B, R, 3, H, W]

def lut_transform(imgs, luts):
    """
    Args:
        imgs: [B, 3, H, W], range [0,1]
        luts: [B, R, 3, m, m, m]
    Returns:
        a_map_list: List of R tensors, each [B, 3, H, W]
    """
    B, C, H, W = imgs.shape
    R = luts.shape[1]
    m = luts.shape[3]
    device = imgs.device

    # Normalize RGB to [0, m-1] and reshape for LUT indexing
    grid = (imgs.permute(0, 2, 3, 1) * (m - 1)).clamp(0, m - 1 - 1e-6)  # [B, H, W, 3]
    x, y, z = grid.unbind(-1)  # Each is [B, H, W]

    x0, y0, z0 = x.floor().long(), y.floor().long(), z.floor().long()
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    x0, x1 = x0.clamp(0, m-1), x1.clamp(0, m-1)
    y0, y1 = y0.clamp(0, m-1), y1.clamp(0, m-1)
    z0, z1 = z0.clamp(0, m-1), z1.clamp(0, m-1)

    xd, yd, zd = x - x0.float(), y - y0.float(), z - z0.float()

    a_map_list = []

    for r in range(R):
        a_r = torch.zeros(B, C, H, W, device=device)

        for c in range(3):
            lut = luts[:, r, c]  # [B, m, m, m]

            # Batch interpolation
            c000 = lut[torch.arange(B)[:, None, None], x0, y0, z0]
            c001 = lut[torch.arange(B)[:, None, None], x0, y0, z1]
            c010 = lut[torch.arange(B)[:, None, None], x0, y1, z0]
            c011 = lut[torch.arange(B)[:, None, None], x0, y1, z1]
            c100 = lut[torch.arange(B)[:, None, None], x1, y0, z0]
            c101 = lut[torch.arange(B)[:, None, None], x1, y0, z1]
            c110 = lut[torch.arange(B)[:, None, None], x1, y1, z0]
            c111 = lut[torch.arange(B)[:, None, None], x1, y1, z1]

            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            out = c0 * (1 - zd) + c1 * zd  # [B, H, W]

            a_r[:, c] = out  #  No permute needed, shape is correct

        a_map_list.append(a_r)  # [B, 3, H, W]

    return a_map_list



class LUT1DGenerator(nn.Module):
    r"""The 1DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points.
        n_feats (int): Dimension of the input image representation vector.
        color_share (bool, optional): Whether to share a single 1D LUT across
            three color channels. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, color_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not color_share else 1
        self.lut1d_generator = nn.Linear(
            n_feats, n_vertices * repeat_factor)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.color_share = color_share

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        lut1d = self.lut1d_generator(x).view(
            x.shape[0], -1, self.n_vertices)
        if self.color_share:
            lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
        lut1d = lut1d.sigmoid()
        return lut1d


class LUT3DGenerator(nn.Module):
    r"""The 3DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks):
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def forward(self, x):
        # weights = self.weights_generator(x)
        # luts = self.basis_luts_bank(weights)
        #
        # luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))

        weights = self.weights_generator(x)  # [B, n_ranks]

        # 返回的是 basis_luts_bank 的 weight（不参与前向传播，只用于后续操作）
        basis_luts = self.basis_luts_bank.weight.T  # [n_ranks, 3*9*9*9]
        basis_luts = basis_luts.view(self.n_ranks, self.n_colors, self.n_vertices, self.n_vertices, self.n_vertices)
        basis_luts = basis_luts.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1, -1)  # [B, n_ranks, 3, 9, 9, 9]
        # print(luts.shape)
        return weights, basis_luts

    # def forward(self, x):
    #     weights = self.weights_generator(x) #产生线性插值的权值
    #     luts = self.basis_luts_bank(weights) #计算偏置
    #     luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors)) #生成最后的线性插值
    #
    #     return weights, luts #权重用于loss

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity