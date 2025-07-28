import os
import argparse
from glob import glob
import numpy as np
from model import R2RNet,APBSN
# from ptflops import get_model_complexity_info
import torch,thop
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--epochs', dest='epochs', type=int, default=20,
                    help='number of total epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2,
                    help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=256,
                    help='patch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--data_dir', dest='data_dir',
                    # default=r'./Training data',
                    default=r'./Training data/SIDD_s256_o128_trainlol',
                    help='directory storing the training data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpts/',
                    help='directory for checkpoints')

args = parser.parse_args()
#
# def debug_lut_training(model, inputs, gt_image, criterion, step=0, save_dir="./debug_output"):
#     """
#     调试LUT学习是否成功。
#
#     Args:
#         model: 含有 basis_luts_bank 的模型
#         inputs: 输入特征（如图像或encoder输出）
#         gt_image: ground truth 图像
#         criterion: 损失函数，如 nn.L1Loss()
#         step: 当前训练step（用于保存文件命名）
#     """
#     os.makedirs(save_dir, exist_ok=True)
#
#     model.eval()
#     with torch.no_grad():
#         # 前向传播
#         weights, luts = model.lut_generator(inputs)  # 假设有 model.lut_generator 结构
#         enhanced = model.apply_lut(inputs, weights, luts)  # 假设你有这个方法
#         loss = criterion(enhanced, gt_image)
#
#         # 输出 loss
#         print(f"[Step {step}] Loss: {loss.item():.4f}")
#
#         # 1️⃣ 检查 LUT 是否有梯度（上一轮训练之后）
#         for name, param in model.named_parameters():
#             if 'basis_luts_bank' in name:
#                 grad = param.grad
#                 if grad is None:
#                     print(f"⚠️ {name} grad is None!")
#                 else:
#                     print(f"✅ {name} grad.norm(): {grad.norm().item():.6f}")
#
#         # 2️⃣ 输出每个 LUT 的值范围
#         print("📊 LUT value stats:")
#         for i in range(min(8, luts.shape[1])):  # B, 8, 3, 9, 9, 9
#             lut = luts[0, i]  # shape: [3, 9, 9, 9]
#             print(f"  LUT {i}: min={lut.min().item():.4f}, max={lut.max().item():.4f}, mean={lut.mean().item():.4f}")
#
#         # 3️⃣ 保存图像前后对比
#         # def to_img(t): return TF.to_pil_image(t.squeeze().clamp(0, 1).cpu())
#         # input_img = to_img(inputs[0])
#         # enhanced_img = to_img(enhanced[0])
#         # gt_img = to_img(gt_image[0])
#         #
#         # input_img.save(os.path.join(save_dir, f"{step:04d}_input.png"))
#         # enhanced_img.save(os.path.join(save_dir, f"{step:04d}_enhanced.png"))
#         # gt_img.save(os.path.join(save_dir, f"{step:04d}_gt.png"))
#         # print(f"🖼️ Saved images at {save_dir}/")
#
#         # 4️⃣ 可视化 LUT 分布（前两个）
#         for i in range(2):
#             lut = luts[0, i].cpu().numpy().transpose(1, 2, 3, 0).reshape(-1, 3)
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(lut[:, 0], lut[:, 1], lut[:, 2], c=lut, s=5)
#             ax.set_title(f"LUT 0-{i}")
#             fig.savefig(os.path.join(save_dir, f"{step:04d}_lut_{i}.png"))
#             plt.close()
#

def train(model):

    lr = args.lr * np.ones([args.epochs])
    lr[10:] = lr[0] / 10.0

    # train_low_data_names = glob(args.data_dir + '/lsrw/*.jpg')
    train_low_data_names = glob(args.data_dir + '/low/*.png')+glob(args.data_dir + '/low/*.jpg')+glob(args.data_dir + '/low/*.bmp')
    train_low_data_names.sort()
    # train_high_data_names = glob(args.data_dir + '/Huawei/high/*.jpg') + \
    #                         glob(args.data_dir + '/Nikon/high/*.jpg')
    # train_high_data_names = glob(args.data_dir + '/high/*.png')+glob(args.data_dir + '/high/*.jpg')+glob(args.data_dir + '/high/*.bmp')
    # train_high_data_names.sort()
    eval_low_data_names = glob('./data/eval15/low/*.png')
    # eval_low_data_names = glob('D:/dierpian/LRSWEval/zong/lrswlow/*.jpg')

    eval_low_data_names.sort()
    eval_high_data_names = glob('./data/eval15/high/*.png')
    # eval_high_data_names = glob('D:/dierpian/LRSWEval/zong/lrswhigh/*.jpg')

    eval_high_data_names.sort()
    # assert len(train_low_data_names) == len(train_high_data_names)
    print('Number of training data: %d' % len(train_low_data_names))

    # M
    # summary = 'macs: %s -->' % (macs / 1e9) + '\n'
    model.train(train_low_data_names,

                eval_low_data_names,
                eval_high_data_names,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=args.epochs,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=1,
                 train_phase="Decom")
    # debug_lut_training(model, input_tensor, gt_tensor, criterion, step=epoch)
    model.train(train_low_data_names,

                eval_low_data_names,
                eval_high_data_names,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=args.epochs,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=1,
                train_phase="Denoise")

    # model.train(train_low_data_names,
    #             train_high_data_names,
    #             eval_low_data_names,
    #             eval_high_data_names,
    #             batch_size=args.batch_size,
    #             patch_size=args.patch_size,
    #             epoch=args.epochs,
    #             lr=lr,
    #             vis_dir=args.vis_dir,
    #             ckpt_dir=args.ckpt_dir,
    #             eval_every_epoch=1,
    #             train_phase="Relight")


if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the checkpoints and visuals
        args.vis_dir = args.ckpt_dir + '/visuals/'
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        model = R2RNet().cuda()
        # Train the model
        train(model)


    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
