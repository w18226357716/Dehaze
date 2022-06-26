from __future__ import print_function
import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataSet_HDF5, DataValSet
from importlib import import_module
import random
import re
import time
import statistics
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from pytorch_ssim.metrics import *
from math import log10

from networks.CR import *


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--step", type=int, default=30, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.1, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument('--dataset', default="", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--dataset1', type=str, default='SOTS', help='Path of the validation dataset')
parser.add_argument('--model', default='net', type=str, help='Import which network')
parser.add_argument('--name', default='', type=str, help='Filename of the training models')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--train_step", type=int, default=1, help="Activated gate module")
parser.add_argument("--clip", type=float, default=0.25, help="Clipping Gradients. Default=0.1")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument('--is_ab', type=bool, default=False)

training_settings=[
    {'nEpochs': 100, 'lr': 1e-4, 'step': 10, 'lr_decay': 0.75}
]


criter = torch.nn.L1Loss()
class perceptualloss(nn.Module):
    def __init__(self):
        super(perceptualloss, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=True).features[0:3]
        self.model.eval()
        self.model.to(device)

    def forward(self, x, y):
        out1 = self.model(x)
        out2 = self.model(y)
        loss = criter(out1, out2)
        return loss

# 获取参数个数
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# 创建文件夹
def mkdir_steptraing():
    root_folder = os.path.abspath('.')    # 返回当前文件的绝对路径（os.path.abspath（）获取文件的绝对路径）
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    step1_folder, step2_folder, step3_folder, step4_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3'), join(models_folder, '4')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        os.makedirs(step4_folder)
        print("===> Step training models store in models/1 & /2 & /3.")

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])

# not in use
def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[-3:-2])
    start_epoch = "".join(re.findall(r"\d", resume)[-2:])
    return int(trainingstep), int(start_epoch)+1
# 动态调整学习率
def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 训练过程中文件的保存
def checkpoint(step, epoch):
    root_folder = os.path.abspath('.')    # 返回当前路径的绝对路径
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    model_out_path = join(models_folder, "{0}/MSBDN_epoch_{1:02d}.pkl".format(step, epoch))
    torch.save(model.module, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

# 训练过程
def train(train_gen, model, criterion, optimizer, epoch):

    epoch_loss = 0
    start_time_data=0
    med_time_data = []
    med_time_gpu = []

    for iteration, batch in enumerate(train_gen, 1):
        evalation_time_data = time.perf_counter() - start_time_data
        med_time_data.append(evalation_time_data)
        start_time_gpu = time.perf_counter()

        Hazy = batch[0]
        GT = batch[1]
        Hazy = Hazy.to(device)
        GT = GT.to(device)

        dehaze1, dehaze2, dehaze3, dehaze4, dehaze5, dehaze6 = model(Hazy)

        loss1 = criterion[0](dehaze1, GT) + criterion[1](dehaze1, GT, Hazy) + ssimloss(dehaze1, GT)
        loss2 = criterion[0](dehaze2, GT) + ssimloss(dehaze2, GT)
        loss3 = criterion[0](dehaze3, GT) + ssimloss(dehaze3, GT)
        loss4 = criterion[0](dehaze4, GT) + ssimloss(dehaze4, GT)
        loss5 = criterion[0](dehaze5, GT) + ssimloss(dehaze5, GT)
        loss6 = criterion[0](dehaze6, GT) + ssimloss(dehaze6, GT)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        evalation_time_gpu = time.perf_counter() - start_time_gpu
        med_time_gpu.append(evalation_time_gpu)

        if iteration % 100 == 0:
            median_time_data = statistics.median(med_time_data)
            median_time_gpu = statistics.median(med_time_gpu)
            print("===> Loading Time: {:.6f}; Runing Time:{:.6f}".format(median_time_data, median_time_gpu))
            print("===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), loss.cpu()))
            med_time_data = []
            med_time_gpu = []
        start_time_data = time.perf_counter()
    print("===>Epoch{} Part: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))
    return epoch_loss / len(trainloader)

def validate(test_gen, model, epoch):
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    # avg_loss0 = 0

    with torch.no_grad():
        for iteration, batch in tqdm(enumerate(test_gen, 1)):
            # print(iteration)
            Blur = batch[0]
            HR = batch[1]
            Blur = Blur.to(device)
            HR = HR.to(device)

            sr, _, _, _, _, _ = model(Blur)

            # sr = model(Blur)
            ssim1 = ssim(sr, HR)
            #print(ssim)
            avg_ssim += ssim1
            # mse = criterion(sr, HR)
            # loss0 = mse
            # psnr = 10 * log10(1 / mse)
            psnr1 = psnr(sr, HR)

            # avg_loss0 = loss + avg_loss0
            avg_psnr += psnr1

        print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / iteration))
        print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
        with open("Logs/Output_{0}.txt".format(opt.name), "a+") as text_file:
            # print("===>Epoch{} Complete: Avg_test:{}\n".format(epoch, avg_loss0 / iteration), file=text_file)
            print("===>Epoch{} Complete: Avg. SR SSIM: {:.4f},Avg. SR PSNR:{:4f} dB\n".format(epoch, avg_ssim / iteration, avg_psnr / iteration), file=text_file)

    model.train()

if __name__ == '__main__':
    opt = parser.parse_args()
    Net = import_module('networks.' + opt.model)
    # print(opt.resume)
    str_ids = list(map(int, opt.gpu_ids.split(',')))
    # print(str_ids[0])
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
    # str_ids = opt.gpu_ids.split(',')
    torch.cuda.set_device(int(str_ids[0]))
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)  # 为特定CPU设置种子，生成随机数（确保每次生成固定的随机数）
    torch.cuda.manual_seed(opt.seed)  # 为特定GPU设置种子，生成随机数

    train_dir = opt.dataset
    root_val_dir = opt.dataset1
    train_sets = [x for x in sorted(os.listdir(train_dir)) if is_hdf5_file(x)]
    print("===> Loading model {} and criterion".format(opt.model))

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("Loading from checkpoint {}".format(opt.resume))
            model = Net.make_model(opt)
            model_dict = model.state_dict()
            print(get_n_params(model))
            pretrained_model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            pretrained_dict = pretrained_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # print(model)
            print(get_n_params(model))
            opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)
            # opt .start_training_step = opt .start_training_step - 1
            print(opt.start_training_step)
            print(opt.start_epoch)
            mkdir_steptraing()
    else:
        model = Net.make_model(opt)
        print(get_n_params(model))
        mkdir_steptraing()


    model = model.to("cuda")
    model = DataParallel(model, device_ids=str_ids, output_device=device)

    criterion = []
    criterion.append(torch.nn.MSELoss(size_average=True).to(device))
    criterion.append(ContrastLoss(ablation=opt.is_ab))
    # criterion.append(perceptualloss())

    # criterion = torch.nn.MSELoss(size_average=True)
    # criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for i in range(opt.start_training_step, 2):
        opt.nEpochs = training_settings[i - 1]['nEpochs']
        opt.lr = training_settings[i - 1]['lr']
        opt.step = training_settings[i - 1]['step']
        opt.lr_decay = training_settings[i - 1]['lr_decay']
        print(opt)
        for epoch in range(opt.start_epoch, opt.nEpochs+1):
            loss = 0
            adjust_learning_rate(epoch)
            random.shuffle(train_sets)
            for j in range(len(train_sets)):
                print("Step {}:Training folder is {}".format(i, join(train_dir, train_sets[j])))
                train_set = DataSet_HDF5(join(train_dir, train_sets[j]))
                trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=1)
                avg_loss = train(trainloader, model, criterion, optimizer, epoch)
                loss = loss + avg_loss
            if epoch % 1 == 0:
                checkpoint(i, epoch)
                testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
                validate(testloader, model, epoch)
            loss = loss / len(train_sets)
            print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, loss))
            with open("Logs/Output_{0}.txt".format(opt.name), "a+") as text_file:
                print("===>Epoch{} Complete: Avg loss is :{:4f}\n".format(epoch, loss), file=text_file)

        opt.start_epoch = 1
