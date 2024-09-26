import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from models.networks import HKD_Student
from dataset import AVADataset
from util import EMD2Loss,EMD1Loss, AverageMeter
import option
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
import clip
from torch import nn

from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))

def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 2 epochs starting from epoch 4"""
    epoch=epoch+1
    if epoch < 4:
        lr = params.init_lr
        print(f"Epoch {epoch}: LR : {lr}")
    else:
        # Check if the number of epochs since 4 is even (i.e., every 2 epochs starting from 4)
        if (epoch - 4) % 2 == 0:
            # Calculate the number of two-epoch intervals since epoch 4
            step = (epoch - 4) // 2
            # Calculate the new learning rate
            lr = params.init_lr * (0.5 ** step)
            print(f"Epoch {epoch}: LR : {lr}")
        else:
            # Maintain the previous learning rate if it's not time to adjust
            lr = optimizer.param_groups[0]['lr']  # Assuming all param_groups share the same lr
            print(f"Epoch {epoch}: LR : {lr}")

    # Update the learning rate in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_score(opt,y_pred):
    w = torch.from_numpy(np.linspace(1,10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_AVA_save_csv, 'train.csv')
    val_csv_path = os.path.join(opt.path_to_AVA_save_csv, 'val.csv')
    test_csv_path = os.path.join(opt.path_to_AVA_save_csv, 'test.csv')

    train_ds = AVADataset(train_csv_path, opt.path_to_AVA_images,  clip.tokenize)
    val_ds = AVADataset(val_csv_path, opt.path_to_AVA_images, clip.tokenize)
    test_ds = AVADataset(test_csv_path, opt.path_to_AVA_images, clip.tokenize)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.train_num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.test_num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.test_num_workers, shuffle=False)
    return train_loader, val_loader,test_loader

def train(opt, model, loader, optimizer, criterion, L2loss, epoch):
    model.train()
    train_losses = AverageMeter()
    x=epoch+1
    k=1
    x0=15
    w =k / (k + np.exp((x - x0) / k))
    for param in model.teacher_model.parameters():
        param.requires_grad = False
    for idx, (image, label, text) in enumerate(tqdm(loader)):
        image = image.to(opt.device)
        label = label.to(opt.device)
        text = text.to(opt.device)
        T_pred, S_pred, relation_t, relation_s, h_t, h_s = model(image, text)

        loss1 = criterion(p_target=label, p_estimate=S_pred)
        Hard_loss=loss1

        loss2 = L2loss(h_t,h_s)
        loss3 = L2loss(relation_t,relation_s)
        loss4 = criterion(p_target=T_pred, p_estimate=S_pred)
        Soft_loss=loss2+loss3+loss4

        loss = Hard_loss + w*Soft_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()  # Update learning rate

        train_losses.update(loss.item(), image.size(0))

    return train_losses.avg

def validate(opt, model, loader, criterion):
    model.eval()
    validate_losses_t = AverageMeter()
    EMD1_losses_t = AverageMeter()
    validate_losses_s = AverageMeter()
    EMD1_losses_s = AverageMeter()
    true_score = []
    pred_score_t = []
    pred_score_s = []
    criterion1 = EMD1Loss()
    criterion1.to(opt.device)

    with torch.no_grad():
        for idx, (image, label, text) in enumerate(tqdm(loader)):
            image = image.to(opt.device)
            label = label.to(opt.device)
            text = text.to(opt.device)

            score_t, score_s, relation_t,relation_s,h_t,h_s = model(image, text)

            pscore_t, pscore_np_t = get_score(opt, score_t)
            pscore_s, pscore_np_s = get_score(opt, score_s)
            pred_score_t += pscore_np_t.tolist()
            pred_score_s += pscore_np_s.tolist()

            # Update true scores list
            tscore, tscore_np = get_score(opt, label)
            true_score += tscore_np.tolist()

            # Calculate losses
            loss_t = criterion(p_target=label, p_estimate=score_t)
            loss_s = criterion(p_target=label, p_estimate=score_s)
            EMD1_t = criterion1(p_target=label, p_estimate=score_t)
            EMD1_s = criterion1(p_target=label, p_estimate=score_s)

            # Update loss meters
            validate_losses_t.update(loss_t.item(), image.size(0))
            validate_losses_s.update(loss_s.item(), image.size(0))
            EMD1_losses_t.update(EMD1_t.item(), image.size(0))
            EMD1_losses_s.update(EMD1_s.item(), image.size(0))

    # Calculate evaluation metrics
    srcc_mean_t, _ = spearmanr(pred_score_t, true_score)
    lcc_mean_t, _ = pearsonr(pred_score_t, true_score)
    rmse_t = np.sqrt(((np.array(pred_score_t) - np.array(true_score)) ** 2).mean(axis=None))
    mse_t = ((np.array(pred_score_t) - np.array(true_score)) ** 2).mean(axis=None)
    mae_t = (abs(np.array(pred_score_t) - np.array(true_score))).mean(axis=None)

    srcc_mean_s, _ = spearmanr(pred_score_s, true_score)
    lcc_mean_s, _ = pearsonr(pred_score_s, true_score)
    rmse_s = np.sqrt(((np.array(pred_score_s) - np.array(true_score)) ** 2).mean(axis=None))
    mse_s = ((np.array(pred_score_s) - np.array(true_score)) ** 2).mean(axis=None)
    mae_s = (abs(np.array(pred_score_s) - np.array(true_score))).mean(axis=None)

    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.0, 0, 1)
    pred_score_t = np.array(pred_score_t)
    pred_score_s = np.array(pred_score_s)
    pred_score_label_t = np.where(pred_score_t <= 5.0, 0, 1)
    pred_score_label_s = np.where(pred_score_s <= 5.0, 0, 1)

    acc_t = accuracy_score(true_score_label, pred_score_label_t)
    acc_s = accuracy_score(true_score_label, pred_score_label_s)

    # print('For score_t:')
    # print('PLCC %4.4f,\tSRCC %4.4f,\tAcc %4.4f,\tEMD1 %4.4f,\tEMD2 %4.4f,\tMSE %4.4f,\tMAE %4.4f,\tRMSE %4.4f' % (
    #     lcc_mean_t, srcc_mean_t, acc_t, EMD1_losses_t.avg, validate_losses_t.avg, mse_t, mae_t, rmse_t))

    # print('For score_s:')
    print('PLCC %4.4f,\tSRCC %4.4f,\tAcc %4.4f,\tEMD1 %4.4f,\tEMD2 %4.4f,\tMSE %4.4f,\tMAE %4.4f,\tRMSE %4.4f' % (
        lcc_mean_s, srcc_mean_s, acc_s, EMD1_losses_s.avg, validate_losses_s.avg, mse_s, mae_s, rmse_s))


    return srcc_mean_s


def start_train(opt):
    train_loader, val_loader, test_loader = create_data_part(opt)
    model = HKD_Student()
    model = model.to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr, eps=1e-3)
    criterion = EMD2Loss()
    L2loss =  nn.MSELoss()

    L2loss.to(opt.device)
    criterion.to(opt.device)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    kl_loss.to(opt.device)
    # scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_epoch, eta_min=opt.min_lr)

    for e in range(opt.num_epoch):
        epoch=e
        adjust_learning_rate(opt, optimizer, e)

        train_loss = train(opt, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, L2loss= L2loss, epoch=epoch)
        srcc_mean = validate(opt, model=model, loader=val_loader, criterion=criterion)
        srcc_mean = validate(opt, model=model, loader=test_loader, criterion=criterion)





if __name__ =="__main__":
    start_train(opt)
