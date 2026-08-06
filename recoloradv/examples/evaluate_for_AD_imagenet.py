import matplotlib.pyplot as plt
import torch
import argparse
import sys
import os
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import datasets, models, transforms
# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st
from recoloradv.mister_ed.utils import pytorch_utils as utils

from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs
import re, glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from us_model import SqueezeNet_ada

def center_crop(t):
    return t[:, :, 16:-16, 16:-16]

parser = argparse.ArgumentParser(
        description='Evaluate a ResNet-50 trained on Imagenet '
        'against ReColorAdv'
    )

parser.add_argument('--imagenet_path', type=str, default='F:/imagenet',
                        help='path to ImageNet dataset')
parser.add_argument('--batch_size', type=int, default=1,
                        help='number of examples/minibatch')
parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
args = parser.parse_args()

normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

device = 'cuda'

net1 = SqueezeNet_ada(3, normalization='None').to(device).eval()
net1.to(device)
pretrained1 = torch.load('F:\defence_code\Adversarial-vertex-mixup-pytorch-main_imagenet_AT_trianboth_onlystyle\ori_imagenet_squeeze_wo_style_ckpt/60.pth')
net1.load_state_dict(pretrained1['net'][0], strict=False)

cw_loss = lf.CWLossF6(net1, normalizer, kappa=float('inf'))
perturbation_loss = lf.PerturbationNormLoss(lp=2)
adv_loss = lf.RegularizedLoss(
    {'cw': cw_loss, 'pert': perturbation_loss},
    {'cw': 1.0, 'pert': 0.05},
    negate=True,
)

pgd_attack = aa.PGD(
    net1,
    normalizer,
    ap.ThreatModel(pt.ReColorAdv, {
        'xform_class': ct.FullSpatial,
        'cspace': cs.CIELUVColorSpace(),
        'lp_style': 'inf',
        'lp_bound': 0.06,
        'xform_params': {
            'resolution_x': 16,
            'resolution_y': 32,
            'resolution_z': 32,
        },
        'use_smooth_loss': True,
    }),
    adv_loss,
)

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # 原始ImageFolder的__getitem__方法只返回(image, label)
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        # 获取图像路径
        path = self.imgs[index][0]
        # 返回图像、标签和路径
        return (original_tuple + (path,))

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])




def ReColorAdv(inputs, labels, model):
    adv_inputs = pgd_attack.attack(
        inputs,
        labels,
        optimizer=optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        signed=False,
        verbose=False,
        num_iterations=(100, 300),
    ).adversarial_tensors()
    # optimizer_kwargs={'lr': 0.001},
    with torch.no_grad():
        adv_logits = model(normalizer(adv_inputs))
        succ = (adv_logits.argmax(1) != labels).detach()
        if succ:
            flag = 1

    return adv_inputs, flag


val_dataset = CustomImageFolder(root='F:/imagenet/val', transform=transform)
# 创建DataLoader
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
all_img_num = 0
attack_succe_img_num = 0

# 初始化进度条
pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Processing Batches")

for batch_idx, (images, labels, paths) in pbar:
    images, labels = images.to(device), labels.to(device)
    for img_idx in range(0, images.shape[0]):
        all_img_num = all_img_num + 1
        adv_img, succ = ReColorAdv(images[img_idx, :, :, :].unsqueeze(0), labels[img_idx].unsqueeze(0), net1)
        if succ == 1:
            attack_succe_img_num = attack_succe_img_num + 1
        original_path = paths[0]
        new_path = original_path.replace('F:/imagenet/val\\', 'F:/imagenet/ReColorAdv/val/')
        new_path = new_path.rsplit('.', 1)[0] + '.npy'

        if not os.path.exists(os.path.dirname(new_path)):
            # 创建文件夹
            os.makedirs(os.path.dirname(new_path))

        succ_acc = attack_succe_img_num / all_img_num

        # 更新进度条描述，包括成功准确率
        pbar.set_description(f"Processing Batches (Success Acc: {succ_acc:.2%})")

        # 保存生成的对抗样本
        np.save(new_path, adv_img.detach().cpu())