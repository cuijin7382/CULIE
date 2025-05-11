import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from . import regist_model

from PIL import Image
class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included.
    see our supple for more details.
    '''

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"
        dim =128
        assert base_ch % 2 == 0, "base channel should be divided with 2"
        # self.bottleneck = IGAB(
        #     dim=dim, dim_head=dim, heads=dim // dim, num_blocks=[2, 4, 4][-1])
        #
        # self.mapping = nn.Conv2d(dim, out_ch, 3, 1, 1, bias=False)
        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        ly = []
        ly += [nn.Conv2d(1, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.headill = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        # self.conv128=nn.Conv2d(36, 128, 1)
        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x,maskillu):
    # def forward(self, x, maskillu):
        x1 = self.head(x)

        # print(illu_fea.shape)
        # illu_fea=self.conv128(illu_fea)
        # x1 = self.bottleneck(x1, illu_fea)
        # maskillu=self.headill(maskillu)
        br1 = self.branch1(x1,maskillu)
        br2 = self.branch2(x1,maskillu)

        x2 = torch.cat([br1, br2], dim=1)
        x3=self.tail(x2)
        return x3


    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

def transform_invert(img_):
    img_ = img_.squeeze(0).transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = img_.detach().cpu().numpy() * 255.0

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        self.head_dark=nn.Sequential(
            CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
            nn.ReLU(inplace=True)
        )
        self.head_bright = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.Conv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)
        )
        # self.tradConv = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch,kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
        #     nn.ReLU(inplace=True)
        # )
        ly = []
        ly += [nn.Conv2d(in_ch * 2, in_ch, kernel_size=1)]


        self.conv1 = nn.Sequential(*ly)

        # ly = []
        #
        # ly += [nn.ReLU(inplace=True)]
        # ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        # ly += [nn.ReLU(inplace=True)]
        # ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        # ly += [nn.ReLU(inplace=True)]
        #
        # ly += [DCl(stride, in_ch) for _ in range(num_module)]
        #
        # ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        # ly += [nn.ReLU(inplace=True)]
        ly = []

        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        # self.body = nn.Sequential(*ly)

        self.body = nn.Sequential(*ly)

        ly = []


        ly += [mDCl(stride, in_ch)]


        # self.body = nn.Sequential(*ly)

        self.mbody = nn.Sequential(*ly)
        # self.R3=True
    def forward(self, x,maskillu):
        # print("bsnx",maskillus.shape)
        # if not self.R3:
        #     xout=self.head_dark(x)+self.head_bright(x*maskillu)
        #
        #     xout=self.body(xout)
        #     print('not false')
        #     return xout
        #
        # else:
        #     xout = self.head_dark(x)
        #     xout = self.body(xout)
        #     print('truebsn r3')
        #     return xout
        # xout = self.head_dark(x) + self.head_bright(x * maskillu)
        # xout = self.head_dark(x) + self.tradConv(x * maskillu)
        xout1 = self.head_dark(x)
        xouu2= self.head_bright(x * maskillu)
        # img_higho = transform_invert(xouu2)
        # # img_higho = transform_invert(torch.mean(input, dim=1, keepdim=True))
        # # im = Image.fromarray(img_high)
        # img_higho.save("xouu2.png")

        xout1=self.body(xout1)
        xouu2=self.mbody(xouu2)
        xout=torch.cat([xout1,+xouu2],dim=1)
        xout=self.conv1(xout)
        # xout=self.body(xout)
        # print('not false')
        return xout

class mDCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, dilation=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=2, dilation=2)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=4, dilation=4)]
        ly += [nn.ReLU(inplace=True)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=2, dilation=2)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, dilation=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)
class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
class IG_MSA(nn.Module): #MSA使用ORF捕获的照明表示来指导自注意的计算。照度引导注意力
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans): #两个接收的
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        # print(x.shape)torch.Size([2, 14400, 128])
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        # print(v.shape)e([2, 1, 14400, 128])
        # print(illu_attn.shapetorch.Size([2, 1, 14400, 36])
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module): #IGT的基本单元是IGAB，它由两层归一化（LN）、一个IG-MSA和一个前馈网络（FFN）组成。(
    def __init__(
            self,
            dim,
            dim_head=128,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads), #注意力
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        #illu_fea用去噪处理之后的特征还是之前的照度特征
        x = x.permute(0, 2, 3, 1) #将张量x的维度顺序进行调整,并将结果存储在一个新的张量 统一到一个维度  归一化
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x #处理注意力
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2) #转回之前维度 二个维度 不懂
        return out


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
