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
from pyiqa import create_metric
import numpy as np
import csv
from PIL import Image
# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs
import re, glob
def check_files_for_string(folder_path, search_string):
    """
    检测文件夹下包含指定字符串的文件

    参数：
    folder_path (str): 文件夹路径
    search_string (str): 要搜索的字符串

    返回：
    包含指定字符串的文件路径列表
    """
    # 使用glob.glob函数获取指定文件夹下所有文件的路径
    all_files = glob.glob(os.path.join(folder_path, "*"))

    # 存储包含指定字符串的文件路径的列表
    result_files = []

    # 遍历所有文件路径
    for file_path in all_files:
        # 使用os.path.basename函数获取文件名
        file_name = os.path.basename(file_path)
        # 检查文件名中是否包含指定的字符串
        if file_name.startswith(search_string):
            result_files.append(file_path)
            return True
    return False

def load_ground_truth(csv_filename):
    image_id_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append('F:\Code\DeepRobust-master\dataset/images/'+row['ImageId']+'.png')
            label_tar_list.append( int(row['TrueLabel'])-1 )

    return image_id_list,label_tar_list
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        # 从数据集中获取一个样本和对应的标签
        sample = self.data[index]
        label = self.labels[index]
        # 读取图像文件
        image = Image.open(sample)
        # 使用预处理操作处理图像样本
        image = self.transform(image)
        return image, label
    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

if __name__ == '__main__':
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

    # model = models.mobilenet_v3_small(pretrained=True, progress=True)
    # arch = 'inception_v3'
    arch = 'mobilenet_v3_small'
    model_file = r'F:\Code\places365-master\ckpt\82mobilenet_v3_small_latest.pth.tar'
    model = models.__dict__[arch](num_classes=365).cuda()
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

    normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])


    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('F:/Code/DeepRobust-master/deeprobust/image/data/ImageNet/val', transforms.Compose([
    #         transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    #     ])), batch_size=1, shuffle=False)
    # image_id_list, label_tar_list = load_ground_truth(r'F:\Code\DeepRobust-master\dataset/images.csv')
    # dataset = CustomDataset(image_id_list, label_tar_list)
    # val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(r'F:\place365\val_only3', transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)
    # 创建 DataLoader，并设置 collate_fn 参数来处理标签
    # val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('/sde_data/xmd_datasets/imagenet/val', transforms.Compose([
    #         transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    #         ])), batch_size=1, shuffle=False)
    # val_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    # )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    cw_loss = lf.CWLossF6(model, normalizer, kappa=float('inf'))
    perturbation_loss = lf.PerturbationNormLoss(lp=2)
    adv_loss = lf.RegularizedLoss(
        {'cw': cw_loss, 'pert': perturbation_loss},
        {'cw': 1.0, 'pert': 0.05},
        negate=True,
    )

    pgd_attack = aa.PGD(
        model,
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

    batches_correct = []
    score_ori_all = []
    score_adv_all = []
    image_succ = 0
    image_num = 0
    transfer_succ_num = 0
#incv3_lr0.01 muq=63.3
    now_save_img_path = r'F:\Code\ReColorAdv-master\mobilenetv3_standard_place365/'
    if not os.path.exists(now_save_img_path):
        # 若不存在，则创建文件夹
        os.makedirs(now_save_img_path)
    for batch_index, (inputs, labels) in enumerate(val_loader):
        # if batch_index < 1351:
        #     image_num = image_num + inputs.shape[0]
        #     continue
        if check_files_for_string(now_save_img_path, str(image_num + 1) + '_ImageNetc_adv'):
            selected_files_name = [file_name for file_name in os.listdir(now_save_img_path) if
                                   file_name.startswith(
                                       str(image_num + 1) + '_ImageNetc_adv') and file_name.endswith(".npy")][0]
            labelis = int(re.search(r"labelis(\d+)", selected_files_name).group(1))
            preis = int(re.search(r"preis(\d+)", selected_files_name).group(1))
            if labelis != preis:
                image_succ = image_succ + 1
            else:
                image_succ = image_succ
            image_num = image_num + 1
            continue

        if (
            args.num_batches is not None and
            batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        adv_inputs = pgd_attack.attack(
            inputs,
            labels,
            optimizer=optim.Adam,
            optimizer_kwargs={'lr': 0.001},
            signed=False,
            verbose=False,
            num_iterations=(100, 300),
        ).adversarial_tensors()
        #optimizer_kwargs={'lr': 0.001},
        with torch.no_grad():
            adv_logits = model(normalizer(adv_inputs))
        batch_correct = (adv_logits.argmax(1) != labels).detach()

        # batch_accuracy = batch_correct.float().mean().item()

        # with torch.no_grad():  # 禁用梯度计算，以提高预测速度和减少内存占用
        #     model2 = models.densenet121(pretrained=True).to('cuda').eval()
        #     predict2 = model2(normalizer(adv_inputs))
        # transfer_succ = (predict2.argmax(1) != labels).detach()
        # transfer_succ = transfer_succ.float().mean().item()

        for img_idx in range(0, adv_inputs.shape[0]):
            image_succ = image_succ + int(batch_correct[img_idx])
            image_num = image_num + 1
            image_succ_rate = image_succ/image_num
            img_tep = adv_inputs[img_idx,:,:,:].unsqueeze(0).cpu().detach().numpy()
            img_tep = img_tep.swapaxes(1, 3).swapaxes(1, 2)[0]

            plt.axis('off')  # 去坐标轴
            plt.xticks([])  # 去刻度
            plt.yticks([])  # 去刻度
            plt.imshow(img_tep)
            np.save(now_save_img_path + str(
                image_num) + '_ImageNetc_adv_succis' + str(int(batch_correct[img_idx])) + '_labelis' +
                    str(int(labels[img_idx])) + '_preis' + str(int(adv_logits.argmax(1)[img_idx])) + '.npy', img_tep)
            plt.savefig(now_save_img_path + str(
                image_num) + '_ImageNetc_adv_succis' + str(int(batch_correct[img_idx])) + '_labelis' +
                    str(int(labels[img_idx])) + '_preis' + str(int(adv_logits.argmax(1)[img_idx])) + '.png', dpi=300,
                        bbox_inches='tight', pad_inches=-0.01)

            inputs_tep = inputs[img_idx,:,:,:].unsqueeze(0).cpu().detach().numpy()
            inputs_tep = inputs_tep.swapaxes(1, 3).swapaxes(1, 2)[0]
            plt.axis('off')  # 去坐标轴
            plt.xticks([])  # 去刻度
            plt.yticks([])  # 去刻度
            plt.imshow(inputs_tep)
            np.save(now_save_img_path + str(
                image_num) + '_ImageNeta_ori_succis' + str(int(batch_correct[img_idx])) + '_labelis' +
                    str(int(labels[img_idx])) + '_preis' + str(int(adv_logits.argmax(1)[img_idx])) + '.npy', inputs_tep)
            plt.savefig(now_save_img_path + str(
                image_num) + '_ImageNeta_ori_succis' + str(int(batch_correct[img_idx])) + '_labelis' +
                    str(int(labels[img_idx])) + '_preis' + str(int(adv_logits.argmax(1)[img_idx])) + '.png', dpi=300,
                        bbox_inches='tight', pad_inches=-0.01)


            # transfer_succ_num = transfer_succ_num + int(transfer_succ[img_idx])
            # transfer_succ_rate = transfer_succ_num / image_num
            #
            # with torch.no_grad():  # 禁用梯度计算，以提高预测速度和减少内存占用
            #     iqa_model = create_metric('pi', metric_mode='NR').to('cuda').eval()
            #     score_ori_tep = iqa_model(inputs[img_idx,:,:,:].unsqueeze(0).clone().detach()).cpu().data.numpy()
            #     score_adv_tep = iqa_model(adv_inputs[img_idx,:,:,:].unsqueeze(0).clone().detach()).cpu().data.numpy()
            #     score_ori_all.append(score_ori_tep)
            #     score_adv_all.append(score_adv_tep)

            print('当前测试了' + str(
                image_num) + '幅图像, 有' + str(int(image_succ)) + '幅图像被成功攻击,成功率为' + str(image_succ_rate)
                  )


    #     print(f'BATCH {batch_index:05d}',
    #           f'accuracy = {batch_accuracy * 100:.1f}',
    #           sep='\t')
    #     batches_correct.append(batch_correct)
    #
    # accuracy = torch.cat(batches_correct).float().mean().item()
    # print('OVERALL    ',
    #       f'accuracy = {accuracy * 100:.1f}',
    #       sep='\t')
