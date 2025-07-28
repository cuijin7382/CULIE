import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
from tqdm import tqdm
import cv2
import lpips
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def np2tensor(n: np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2, 0, 1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2, 0, 1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s' % (n.shape,))


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def lpips2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    lpips_model = lpips.LPIPS(net="alex").to(device)  # alex
    # numpt to tensor
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # img1 = img1.transpose(1, 2, 0)
    # print(img1.shape) (600, 3, 400)
    img1 = np2tensor(img1).to(device)
    img2 = np2tensor(img2).to(device)
    # print(img1.shape) torch.Size([400, 600, 3])

    # img1 = Image.fromarray(np.uint8(img1))
    # img2 = Image.fromarray(np.uint8(img2))
    # img1 = preprocess(img1).unsqueeze(0).to(device)
    # img2 = preprocess(img2).unsqueeze(0).to(device)

    distance = lpips_model(img1, img2)
    return distance.item()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# loe
def img_loe(ipic, epic, window_size=7):
    def U_feature(image):
        image = cv2.resize(image, (500, 500))
        image = np.max(image, axis=2)
        w_half = window_size // 2
        padded_arr = np.pad(image, ((w_half, w_half), (w_half, w_half)), mode='constant')

        local_windows = np.lib.stride_tricks.sliding_window_view(padded_arr, (window_size, window_size))
        local_windows = local_windows.reshape(-1, window_size * window_size)
        relationship = local_windows[:, :, None] > local_windows[:, None, :]
        return relationship.flatten()

    ipic = U_feature(ipic)
    epic = U_feature(epic)

    return np.mean(ipic != epic)


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


def metrics(im_dir, label_dir):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_loe = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()

    # for item in tqdm(sorted(glob.glob(im_dir))):
    for idx in range(len(glob.glob(im_dir))):
        print(idx)
        n += 1
        # im1 = Image.open(item).convert('RGB')
        dir = glob.glob(im_dir)
        im1 = Image.open(dir[idx]).convert('RGB')
        # name = item.split('/')[-1]
        # name = name.split('_')[0] + '.JPG'              # for SICE
        # im2 = Image.open(label_dir+name).convert('RGB')
        im_dir33 = glob.glob(label_dir)
        im2 = Image.open(im_dir33[idx]).convert('RGB')
        (h, w) = im2.size
        im1 = im1.resize((h, w))
        im1 = np.array(im1)
        im2 = np.array(im2)
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)
        score_loe = img_loe(im1, im2)
        score_lpips = lpips2(im2, im1)
        #
        # ex_p0 = lpips.im2tensor(cv2.resize(lpips.load_image(item), (h, w)))
        # ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name))
        # ex_p0 = ex_p0.cuda()
        # ex_ref = ex_ref.cuda()
        # score_lpips = loss_fn.forward(ex_ref, ex_p0)

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips
        avg_loe += score_loe

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    avg_loe = avg_loe / n
    return avg_psnr, avg_ssim, avg_lpips, avg_loe


if __name__ == '__main__':
    # im_dir ='D:/dierpian/shijuexiaoguoduibi/zero-dce/vv/*.*'
    im_dir = 'D:/lunwen2/newR2RNet-main/output/deno/low/20.57/*.*'
    # im_dir='D:/lunwen2/newR2RNet-main/results/test/low/low/19.87/illu/*.*'
    # im_dir = 'D:/lunwen2/Zero-DCE_extension-main/Zero-DCE++/data/result_Zero_DCE++/real/*.jpg'

    # gt
    label_dir = 'D:/lunwen2/newR2RNet-main/data/eval15/high/*.*'
    # label_dir='D:/dierpian/testdatasets/mef/*.*'
    # label_dir = 'D:/dierpian/LRSWEval/zong/lrswhigh/*.*'
    # im_dir = 'results/SICE/I/*.JPG'
    # label_dir = '../dataset/LIE/SICE-test/label/'

    avg_psnr, avg_ssim, avg_lpips, avg_loe = metrics(im_dir, label_dir)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.loe {:.4f}".format(avg_loe))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
# D:\dierpian\cvpr\PairLIE-main\results\lol\I