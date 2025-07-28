import glob
# from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import imquality.brisque as brisque
from niqe_utils import *
import argparse
import lpips
import torchvision.transforms as transforms
eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
ep = eval_parser.parse_args()
def np2tensor(n:np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s'%(n.shape,))

def ssim2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # if len(img1.shape) == 4:
    #     img1 = img1[0]
    # if len(img2.shape) == 4:
    #     img2 = img2[0]
    #
    # # tensor to numpy
    # if isinstance(img1, torch.Tensor):
    #     img1 = tensor2np(img1)
    # if isinstance(img2, torch.Tensor):
    #     img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)

    return structural_similarity(img1, img2, multichannel=True, data_range=255)
def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def lpips2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    lpips_model = lpips.LPIPS(net="vgg").to(device)#alex
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
def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
    avg_lpips = 0
    for item in tqdm(sorted(glob.glob(im_dir))):
    # for idx in range(len('./data/test/low')):
    #     print(range(len('./data/test/low')))

        n += 1
        # dir = glob.glob(im_dir)
        # im1 = Image.open(dir[idx]).convert('RGB')
        # test_img_path = dir[idx]
        # print(test_img_path)
        # dir=glob.glob(test_img_path)

        # im1 = np.array(im1, dtype="float32") 之前有
        # print(im1)
        # test_img_path = Image.open(test_img_path[idx])
        # print(idx)
        # im1 = np.array(im1, dtype="float32")
        # im1anme = dir[idx].split('/')[-1]
        # print('Processing ', im1anme)
        # test_low_img = Image.open(test_img_path)
        # im1 = np.array(test_low_img, dtype="float32") / 255.0
        # test_low_img = np.transpose(test_low_img, (2, 0, 1))
        # im1 = np.expand_dims(test_low_img, axis=0)
        # print(im1.shape)
        im1 = Image.open(item).convert('RGB')
        # print(im1)
        score_brisque = brisque.score(im1)

        im2 = np.array(im1)  # print(im1)
        # print(im1.shape)(789, 1039, 3)
        score_niqe = calculate_niqe(im2)

        avg_niqe += score_niqe

        # brisque.score(img)



        avg_brisque += score_brisque


        #########lpips
        # im_dir33= glob.glob(im_dir3)
        # eval_high_img = Image.open(im_dir33[idx])
        # # eval_high_img = np.array(eval_high_img, dtype="float32")
        # im3 = np.array(eval_high_img, dtype="float32")
        # score_lpips = lpips2(im2,im3)
        # avg_lpips += score_lpips





        torch.cuda.empty_cache()

    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe /  n
    # avg_lpips = avg_lpips /  n
    return avg_niqe, avg_brisque


if __name__ == '__main__':

    if ep.DICM:
        # im_dir = 'D:/lunwen2/newR2RNet-main/output/illu/LIME/*.png'
        im_dir = 'D:/lunwen2/newR2RNet-main/output/illu/xx/*.png'
    elif ep.LIME:
        im_dir = './output/LIME/*.bmp'

    elif ep.MEF:
        im_dir = './output/MEF/*.png'

    elif ep.NPE:
        im_dir = './output/NPE/*.jpg'

    elif ep.VV:
        im_dir = './output/VV/*.jpg'
    # avg_niqe, avg_brisque = metrics('D:/dierpian/shijuexiaoguoduibi/sci/vv/*.*')

    # avg_niqe, avg_brisque = metrics('D:/lunwen2/newR2RNet-main/output/deno/vv/*.*')     #our:
    avg_niqe, avg_brisque = metrics('D:/dierpian/cvpr/CLIP-LIT-main/inference_result/un/vv/*.*')
    #                                            'D:/lunwen2/newR2RNet-main/data/eval15/high/*.png') #sci:

    #
    # avg_niqe,avg_brisque= metrics('D:/lunwen2/newR2RNet-main/output/illu/mef-32.95/*.*')
    # avg_niqe,avg_brisque= metrics('D:/lunwen2/newR2RNet-main/output/illu-0.7in+0.3put/mef-32.01/*.*')

    # avg_niqe,avg_brisque = metrics('D:/lunwen2/newR2RNet-main/output/deno/DICM/*.JPG')
    # avg_niqe,avg_brisque = metrics('D:/dierpian/shijuexiaoguoduibi/zero-dce++/lime/*.*')
    # avg_niqe, avg_brisque = metrics(im_dir)
    # avg_niqe, avg_brisque = metrics('D:/lunwen2/newR2RNet-main/output/illu/VV/*.JPG')
    print('平均niqe',avg_niqe)
    print('平均bri',avg_brisque)
    # print('平均lpips',avg_lpips)
#