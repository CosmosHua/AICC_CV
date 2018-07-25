# coding=utf8
#########################################################################
# File Name: main.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Sat 03 Feb 2018 03:50:30 PM CST
#########################################################################

import sys
import os
import time
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import threading
import traceback

from tools import parse, py_op, measures, utils, plot
from model import dataset, unet_models, layers, classify

reload(sys)
sys.setdefaultencoding('utf8')
args = parse.args

args.save_dir = args.save_dir + '_' + str(args.input_filter)

try:
    os.mkdir(os.path.join(args.data_dir, 'model'))
    os.mkdir(os.path.join(args.data_dir, 'model', args.save_dir))
except:
    pass

logfile = os.path.join(args.data_dir, 'model', args.save_dir, 'log')
py_op.mkdir(os.path.join(args.data_dir, 'model', args.save_dir))
sys.stdout = utils.Logger(logfile)


def main():

    train_filelist  = glob.glob(os.path.join(args.data_dir, 'AI/trainB/', '*'))# [:1000]
    # train_filelist += glob.glob(os.path.join(args.data_dir, 'celebA/trainB/', '*'))[:1000]
    test_filelist = glob.glob(os.path.join(args.data_dir, 'AI/testB/', '*'))# [:1000]
    # test_filelist += glob.glob(os.path.join(args.data_dir, 'celebA/testB/', '*'))[:1000]
    # train_filelist  = train_filelist[:100]
    # test_filelist   = test_filelist[:10]

    train_dataset = dataset.DataBowl(train_filelist, phase='train')
    pred_dataset = dataset.DataBowl(train_filelist, phase='pred')
    test_dataset = dataset.DataBowl(test_filelist, phase='validation')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=False)
    pred_loader = DataLoader(pred_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=False)
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True)

    net_G = unet_models.AlbuNet(num_classes=4, input_filter=args.input_filter)
    net_D = classify.ResNet([1])
    net_G = torch.nn.DataParallel(net_G.cuda())
    net_D = torch.nn.DataParallel(net_D.cuda())
    best_psnr = [0,1,0]
    best_mask = [0,0]
    gan_acc_train = 0
    if len(args.resume):
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        try:
            net_D.load_state_dict(checkpoint['state_dict_D'])
            best_mask = checkpoint['best_mask']
            best_psnr = checkpoint['best_psnr']
            gan_acc_train = checkpoint['gan_acc_train']
            print 'best mask : epoch:{:d}  mask_acc:{:3.4f}'.format(best_mask[0], best_mask[1])
            print 'best psnr : epoch:{:d}  psnr:{:3.4f} merge_psnr:{:3.4f}'.format(best_psnr[0], best_psnr[1], best_psnr[2])
        except:
            if len(best_psnr) == 2:
                best_psnr.append(0)
            traceback.print_exc()
        # args.save_dir = checkpoint['save_dir']
        # print checkpoint.keys()
        net_G.load_state_dict(checkpoint['state_dict'])
        print 'load epoch ', start_epoch
    else:
        start_epoch = 0
    # best_psnr = [0,1,0]
    # best_mask = [0,0]
    # gan_acc = 0
    net_state = [best_psnr, best_mask, gan_acc_train]


    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=args.lr)
    loss = layers.Loss()
    loss = loss.cuda()

    def get_lr(epoch):
        if epoch <= args.epochs * 0.2:
            lr = args.lr
        elif epoch <= args.epochs * 0.5:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.75:
            lr = 0.01 * args.lr
        else:
            lr = 0.001 * args.lr
        return lr

    for epoch in range(start_epoch, args.epochs):
        if args.test:
            args.write_pred = 1
            net_state = train_eval(test_loader, net_G, net_D, loss, epoch, optimizer_G, optimizer_D, get_lr, net_state, phase='write validation')
            break
        elif args.write_pred == 1:
            net_state = train_eval(pred_loader, net_G, net_D, loss, epoch, optimizer_G, optimizer_D, get_lr, net_state, 'write')
            net_state = train_eval(test_loader, net_G, net_D, loss, epoch, optimizer_G, optimizer_D, get_lr, net_state, phase='write validation')
            break
        else:
            net_state = train_eval(pred_loader, net_G, net_D, loss, epoch, optimizer_G, optimizer_D, get_lr, net_state)
            net_state = train_eval(test_loader, net_G, net_D, loss, epoch, optimizer_G, optimizer_D, get_lr, net_state, phase='validation')
        # break



def train_eval(data_loader, net_G, net_D, loss, epoch, optimizer_G, optimizer_D, get_lr, net_state, phase='train'):
    best_psnr, best_mask, gan_acc_train = net_state 
    print 
    print phase
    print 'gan_acc_train', gan_acc_train

    if args.test or args.write_pred == 1:
        lr = 0
        net_G.eval()
        net_D.eval()
    elif phase == 'train':
        net_G.train()
        net_D.train()
        lr = get_lr(epoch)
    else:
        lr = 0
        net_G.eval()
        net_D.eval()
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr
    loss_list = []

    mask_metric_list = []
    gan_metric_list = []
    print 'epoch', epoch
    for i, data in enumerate(tqdm(data_loader)):
        # return
        # if i > 20:
        #     break

        # file_names = data[-1]
        # data = [Variable(x.cuda(async=True)) for x in data[:-1]]
        clean, stain, mask, file_names, clean_label, stain_label = data
        clean = Variable(clean.cuda(async=True))
        stain = Variable(stain.cuda(async=True))
        mask = Variable(mask.cuda(async=True))
        clean_label = Variable(clean_label.cuda(async=True))
        stain_label = Variable(stain_label.cuda(async=True))
        
        if args.input_filter == 7:
            output = net_G(stain)
        else:
            if args.training_time == 1:
                output = net_G(stain[:,:3,:,:])
            else:
                output = net_G(stain[:,3:6,:,:])
        if args.input_filter == 3 and args.training_time == 2:
            pred_mask = stain[:,6:,:,:].contiguous()
        else:
            pred_mask = output[:,:1,:,:].contiguous()
        pred_clean = output[:,1:,:,:].contiguous()
        
        mask_size = list(pred_mask.size())
        mask_size[1] = 3
        pred_mask_expand = pred_mask.expand(mask_size)
        pred_mask_expand = torch.sigmoid(pred_mask_expand)
        pred_merge = stain[:,:3,:,:] * (1 - pred_mask_expand) + pred_clean * pred_mask_expand
        mask_expand = mask.view(pred_mask.size()).expand(mask_size).data.cpu().numpy()
        mask_expand[mask_expand<0] = 0
        mask_expand = Variable(torch.from_numpy(mask_expand).cuda(async=True))
        
        merge_clean = stain[:,:3,:,:] * (1 - mask_expand) + pred_clean * mask_expand

        if args.use_gan:
            clean_pred_D = net_D(clean)
            stain_pred_D = net_D(pred_clean)

        if phase == 'train':
            if args.use_gan:
                # loss_output = loss(pred_mask, pred_clean, mask, clean, clean_pred_D, stain_pred_D, clean_label, stain_label)
                loss_output = loss(pred_mask, pred_merge, mask, clean, clean_pred_D, stain_pred_D, clean_label, stain_label)
            else:
                loss_output = loss(pred_mask, pred_clean, pred_merge, mask, clean) #, clean_pred_D, stain_pred_D, clean_label, stain_label)
            if not args.use_gan or best_psnr[1] < 31 or gan_acc_train > 0.9 or (gan_acc_train < 0.7 and i % 10 > 0):
                optimizer_G.zero_grad()
                if args.training_time == 1:
                    (loss_output[0] + loss_output[1]).backward()
                else:
                    loss_output[1].backward()
                optimizer_G.step()
            elif gan_acc_train > 0.7 or i % 10 or loss_output[3].data[0]<0.5:
                optimizer_G.zero_grad()
                (loss_output[0] + loss_output[1] + loss_output[2]).backward()
                optimizer_G.step()
            else:
                print '优化D'
                optimizer_D.zero_grad()
                loss_output[3].backward()
                optimizer_D.step()

        if i % 10 == 0 or phase != 'train':
            # 保存中间结果
            middle_result = os.path.join(args.data_dir, 'middle_result')
            if not os.path.exists(middle_result):
                os.mkdir(middle_result)
            middle_result = os.path.join(middle_result, phase)
            if not os.path.exists(middle_result):
                os.mkdir(middle_result)

            batch_size = clean.data.cpu().numpy().shape[0]
            if phase == 'train':
                ii_list = [(i/10)%batch_size]
            else:
                ii_list = [(i/10)%batch_size]
                # ii_list = range(batch_size)
                # np.random.shuffle(ii_list)
                # ii_list = ii_list[:int(len(ii_list)/11)+1]

            def comput_psnr(a,b):
                a = (a * 100 + 128).astype(np.uint8)
                b = (b * 100 + 128).astype(np.uint8)
                return measures.psnr(a,b)

            pred_psnr = comput_psnr(pred_clean.data.cpu().numpy(), clean.data.cpu().numpy())
            merge_psnr = comput_psnr(pred_merge.data.cpu().numpy(), clean.data.cpu().numpy())
            merge_mask = comput_psnr(merge_clean.data.cpu().numpy(), clean.data.cpu().numpy())
            if phase != 'train':
                loss_list.append([pred_psnr, merge_psnr, merge_mask])

            # 保存中间结果
            for ii in ii_list:
                # print ii
                clean_ii = (clean.data.cpu().numpy()[ii] * 100 + 128).astype(np.uint8)
                pred_clean_ii = (pred_clean.cpu().data.numpy()[ii] * 100  + 128).astype(np.uint8)
                stain_ii = (stain.data.cpu().numpy()[ii][:3] * 100  + 128).astype(np.uint8)
                del_stain_ii = (stain.data.cpu().numpy()[ii][3:6] * 100  + 128).astype(np.uint8)
                last_mask_ii = (stain.data.cpu().numpy()[ii][6] * 256).astype(np.uint8)
                last_mask_ii = np.array([last_mask_ii, last_mask_ii, last_mask_ii]).astype(np.uint8)
                mask_ii = (mask.data.cpu().numpy()[ii] > 0.5).astype(np.uint8)
                mask_ii = (np.array([mask_ii,mask_ii,mask_ii]) * 255).astype(np.uint8)
                pred_mask_ii =  torch.sigmoid(pred_mask.data.cpu()[ii,0]).numpy()
                pred_mask_ii = np.array([pred_mask_ii,pred_mask_ii,pred_mask_ii])
                merge_clean_ii = ((1 - pred_mask_ii)* stain_ii + pred_mask_ii * pred_clean_ii).astype(np.uint8)
                pred_mask_ii = (pred_mask_ii * 256).astype(np.uint8)

                if ii == (i/10)%batch_size:
                    Image.fromarray(del_stain_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_2_stain_del.png'.format(ii)))
                    Image.fromarray(clean_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_1_clean.png'.format(ii)))
                    Image.fromarray(pred_clean_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_3_pred.png'.format(ii)))
                    Image.fromarray(merge_clean_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_4_merge.png'.format(ii)))
                    Image.fromarray(stain_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_5_stain.png'.format(ii)))
                    Image.fromarray(last_mask_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_6_mask_last.png'.format(ii)))
                    Image.fromarray(mask_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_7_mask.png'.format(ii)))
                    Image.fromarray(pred_mask_ii.transpose(1,2,0)).save(os.path.join(middle_result,'{:d}_8_mask_pred.png'.format(ii)))

        # 统计 Loss
        if phase == 'train':
            loss_data = []
            for x in loss_output:
                try:
                    loss_data.append(x.data[0])
                except:
                    loss_data.append(0)
            loss_data.append(pred_psnr)
            loss_data.append(merge_psnr)
            loss_data.append(merge_mask)
            loss_list.append(loss_data)

        # BCELoss 下的统计mask结果
        m = measures.stati_class_number_true_flase_bce( mask.data.cpu().numpy(), pred_mask.data.cpu().numpy()>0)
        mask_metric_list.append(m)

        # BCELoss 下的统计discriminator分类结果
        if args.use_gan:
            real_img_label = np.array([clean_label.data.cpu().numpy().reshape(-1),stain_label.data.cpu().numpy().reshape(-1)])
            real_img_pred  = np.array([clean_pred_D.data.cpu().numpy().reshape(-1),stain_pred_D.data.cpu().numpy().reshape(-1)])
            m = measures.stati_class_number_true_flase_bce(real_img_label, real_img_pred)
            gan_metric_list.append(m)

        # 保存所有预测图片
        if args.write_pred > -1:
            if epoch % args.save_pred_freq == args.save_pred_freq or args.write_pred==1:
                # t = threading.Thread(target=write_pred_clean, args=(file_names,pred_clean.data.cpu().numpy(), stain.data.cpu().numpy(), pred_mask.data.cpu(), best_psnr, epoch, merge_psnr))
                t = threading.Thread(target=write_pred_clean, args=(file_names,pred_merge.data.cpu().numpy(), stain.data.cpu().numpy(), pred_mask.data.cpu(), best_psnr, epoch, merge_psnr))
                t.start()

    mask_metric = measures.measures(mask_metric_list)
    # gan_metric = measures.measures(gan_metric_list)
    print 'phase:', phase
    print 'epoch:', epoch
    if phase == 'train':
        loss_list = np.array(loss_list).mean(0)
        # pred_psnr, merge_psnr = np.mean(loss_list[:,-2]),np.mean(loss_list[:,-1])
        print 'mask loss:{:3.2f}\t mse loss: {:3.6f}\t fake loss:{:3.6f}\t net_D_loss:{:3.6f} pred_psnr:{:3.2f}\t merge_psnr:{:3.2f} \t merge_mask: {:3.2f}'.format(
                loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5], loss_list[6])
    else:
        loss_list = np.array(loss_list)
        pred_psnr, merge_psnr, merge_mask = np.mean(loss_list,0)
        print 'pred_psnr:{:3.2f}\t merge_psnr: {:3.2f} \t merge_mask: {:3.2f}'.format(pred_psnr, merge_psnr, merge_mask)
    mask_acc = measures.print_measures(mask_metric, 'mask measures')
    if args.use_gan:
        gan_acc = measures.print_measures(gan_metric, 'GAN measures')
    else:
        gan_acc = 0
    if phase == 'train':
        gan_acc_train = gan_acc

    if (epoch % args.save_freq == 0 and phase == 'train') or phase != 'train':


        state_dict = net_G.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        state_dict_D = net_D.state_dict()
        for key in state_dict_D.keys():
            state_dict_D[key] = state_dict_D[key].cpu()
        state_dict_all = {
                    'epoch': epoch,
                    'save_dir': args.save_dir,
                    'state_dict': state_dict,
                    'state_dict_D': state_dict_D,
                    'args': args,
                    'best_mask': best_mask,
                    'best_psnr': best_psnr,
                    'gan_acc_train': gan_acc_train,
                }

        if args.test or args.write_pred == 1:
            # 不保存
            pass
        elif phase == 'train':
            save_dir = os.path.join(args.data_dir, 'model', args.save_dir, str(epoch))
            if not os.path.exists(save_dir):
                py_op.mkdir(os.path.join(args.data_dir, 'model'))
                py_op.mkdir(os.path.join(args.data_dir, 'model', args.save_dir))
                py_op.mkdir(os.path.join(args.data_dir, 'model', args.save_dir, str(epoch)))
            torch.save( state_dict_all , os.path.join(save_dir, '%03d.ckpt' % epoch))
        else:
            if mask_acc > best_mask[1]:
                save_dir = os.path.join(args.data_dir, 'model', args.save_dir, 'best_mask')
                if not os.path.exists(save_dir):
                    py_op.mkdir(save_dir)
                best_mask = [epoch, mask_acc]
                state_dict_all['best_mask'] = best_mask
                torch.save( state_dict_all , os.path.join(save_dir, 'best_mask_{:d}.ckpt').format(args.input_filter)) 
            if max(pred_psnr, merge_psnr) > max(best_psnr[1:]):
                save_dir = os.path.join(args.data_dir, 'model', args.save_dir, 'best_psnr')
                if not os.path.exists(save_dir):
                    py_op.mkdir(save_dir)
                best_psnr = [epoch, pred_psnr, merge_psnr]
                state_dict_all['best_psnr'] = best_psnr
                torch.save(state_dict_all, os.path.join(save_dir, 'best_psnr_{:d}.ckpt').format(args.input_filter)) 
        print 'best mask : epoch:{:d}  mask_acc:{:3.4f}'.format(best_mask[0], best_mask[1])
        print 'best psnr : epoch:{:d}  pred psnr:{:3.4f} merge psnr:{:3.4f}'.format(best_psnr[0], best_psnr[1], best_psnr[2])
    print

    return best_psnr, best_mask, gan_acc_train

def write_pred_clean(file_names, pred_clean, stain, pred_mask, best_psnr, epoch, merge_psnr):
    if 'tmp' in args.save_dir:
        return

    pred_mask_init = pred_mask
    pred_mask = torch.sigmoid(pred_mask).numpy()
    pred_mask = np.concatenate([pred_mask,pred_mask,pred_mask], 1)

    pred_clean = (pred_clean * 100 + 128).astype(np.uint8)
    save_data = pred_clean

    pred_mask = (pred_mask * 256).astype(np.uint8)

    if args.test == 1:
        if 'best' in args.resume:
            clean_folder = 'test_clean/best'
            mask_folder = 'test_mask/best'
        else:
            clean_folder = 'test_clean/{:d}_{:d}'.format(args.input_filter, epoch)
            mask_folder = 'test_mask/{:d}_{:d}'.format(args.input_filter, epoch)
    else:
        clean_folder = 'pred_clean'
        mask_folder = 'pred_mask'

    psnr_list = []
    for file_name, save_clean, mask, mask_npy in zip(file_names, save_data, pred_mask, pred_mask_init):
        pred_file_name = file_name.replace(args.data_dir.strip('/'), os.path.join(args.data_dir, clean_folder)).replace('jpg','png')
        folder = '/'.join(pred_file_name.split('/')[:-1])
        if not os.path.exists(folder):
            py_op.mkdir(folder)

        pred_mask_name = file_name.replace(args.data_dir.strip('/'), os.path.join(args.data_dir, mask_folder)).replace('jpg','png')
        folder = '/'.join(pred_mask_name.split('/')[:-1])
        if not os.path.exists(folder):
            py_op.mkdir(folder)

        Image.fromarray(save_clean.transpose(1,2,0)).save(pred_file_name)
        Image.fromarray(mask.transpose(1,2,0)).save(pred_mask_name)
        np.save(pred_mask_name.replace('.png','.npy'), mask_npy)

        continue
        clean_file = pred_file_name.replace(clean_folder+'/','').replace('.png','.jpg')
        clean = np.array(Image.open(clean_file)).astype(np.float32)
        save_new = np.array(Image.open(pred_file_name)).astype(np.float32)
        psnr_pre = measures.psnr(save_clean.transpose(1,2,0).astype(np.float32), clean)
        psnr_new = measures.psnr(save_new, clean)
        psnr_list.append([psnr_pre, psnr_new])
    # psnr = np.array(psnr_list).mean(0)
    # print merge_psnr, psnr[0], psnr[1]


def test(data_loader, net_G, save_dir, config, save_fstr):
    net_G.eval()


if __name__ == '__main__':
    main()
