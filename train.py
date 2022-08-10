
import argparse
import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm
from datetime import datetime
import shutil
import sys

from utils.dataset import HorseDataset,BaseTransform,AccHorseDataset,AugTransform
from utils.logger import setup_logger
from utils.metric import SegmentationMetric
from model import DenseASPP, FastSCNN


def get_args():
    parser = argparse.ArgumentParser()
    # model and dataset
    parser.add_argument('--dataset',type=str,default='accHorse',choices=['horse','cityspaces','accHorse'],
                        help='which dataset to use')
    parser.add_argument('--data_dir',type=str,default='data',
                        help='data dir')
    parser.add_argument('--model',type=str,default='FastSCNN',choices=['DenseASPP','FastSCNN'],
                        help='model name')
    parser.add_argument('--save_dir',type=str,default='saved',
                        help='path to save the model and log')
    parser.add_argument('--aug_data',type=bool,default=False,
                        help='whether to aug the data')
    # training hyper params 
    parser.add_argument('--batch_size',type=int,default=8,
                        help='batch size for training')
    parser.add_argument('--epochs',type=int,default=80,
                        help='run how many epochs')
    parser.add_argument('--lr',type=float,default=1e-2,
                        help='init lr')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum in SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay ratio')
    parser.add_argument('--step_size',type=int,default=20,
                        help='after how many epochs to decay the lr once')
    parser.add_argument('--val_interval',type=int,default=2,
                        help='validation interval')
    parser.add_argument('--save_interval',type=int,default=10,
                        help='model save interval')
    parser.add_argument('--use_amp',type=bool,default=True,
                        help='whether to use mix percision training')
    args = parser.parse_args()
    return args


class Trainer():
    def __init__(self,args):
        self.args = args
        self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        self.transforms = BaseTransform(resize_size = [512,512])
        if args.dataset == 'cityspaces':
            train_dataset = datasets.Cityscapes(args.data_dir+'/cityspaces', split='train', mode='fine',
                                        target_type='semantic',transforms=self.transforms)
            test_dataset = datasets.Cityscapes(args.data_dir+'/cityspaces', split='test', mode='fine',
                                        target_type='semantic',transforms=self.transforms)
            self.nclass = 30
        elif args.dataset == 'horse':
            dataset = HorseDataset(args.data_dir+'/horse_data',self.transforms)
            data_num = len(dataset)
            train_num = int(data_num*0.85)
            train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_num,data_num-train_num])
            self.nclass = 1
        elif args.dataset == 'accHorse':
            dataset = AccHorseDataset(args.data_dir+'/accHorse')
            data_num = len(dataset)
            train_num = int(data_num*0.85)
            train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_num,data_num-train_num])
            self.nclass = 1

        self.train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=4)
        self.test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=4)

        if args.aug_data:
            self.aug_transform = AugTransform()

        if args.model.lower() == 'denseaspp':
            from torchvision.models import DenseNet121_Weights
            weights = DenseNet121_Weights.DEFAULT
            self.model = DenseASPP(nclass=self.nclass,pretrained_weights=weights)
        elif args.model.lower() == 'fastscnn':
            self.model = FastSCNN(self.nclass)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=args.step_size,gamma=0.1,verbose=True)

        self.metric = SegmentationMetric(nclass=self.nclass)

        isDebug = True if sys.gettrace() else False
        if isDebug:
            args.save_dir = 'debug_log'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = args.save_dir+'/'+timestamp+'/log'
        self.model_path = args.save_dir+'/'+timestamp+'/model'
        self.logger = setup_logger(save_dir = self.log_path)
        os.makedirs(self.model_path,exist_ok=True)

        self.logger.info(vars(args))
        self.best_pred = 0

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)


    def train(self):
        self.logger.info('start training')

        self.model.train()
        self.model.to(self.device)
        for e in tqdm(range(self.args.epochs)):
            for i,data in enumerate(self.train_dataloader):
                imgs,targets = data
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                if self.args.aug_data:
                    imgs,targets = self.aug_transform(imgs,targets)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs,targets)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if e % self.args.val_interval == 0 :
                self.validate(e)
        
            self.lr_scheduler.step()

        self.validate(e)# validate once at last
        self.save_checkpoint(e)

    def validate(self,epoch):
        self.metric.reset()
        self.model.eval()
        for i,data in enumerate(self.test_dataloader):
            imgs,targets = data
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                with torch.no_grad():
                    outputs = self.model(imgs)
            self.metric.update(outputs,targets)

            if i == 0:
                save_result(imgs,outputs,targets,self.log_path+'/%i.jpg'%epoch,self.nclass)
        
        pixAcc, mIoU, bIoU = self.metric.get()
        self.logger.info('epoch:%i,pixAcc:%f,mIoU:%f,bIoU:%f'%(epoch,pixAcc,mIoU,bIoU))
        pred = (pixAcc+mIoU)/2
        if epoch % self.args.save_interval == 0:
            self.save_checkpoint(epoch,pred>self.best_pred)
        if pred > self.best_pred:
            self.best_pred = pred

    def save_checkpoint(self,epoch,is_best=False):
        filename = '%s_%s_%s.pth'%(self.args.model,self.args.dataset,epoch)
        filename = os.path.join(self.model_path,filename)
        torch.save(self.model.state_dict(),filename)
        if is_best:
            best_filename = '%s_%s_best.pth'%(self.args.model,self.args.dataset)
            best_filename = os.path.join(self.model_path,best_filename)
            shutil.copyfile(filename,best_filename)

def save_result(imgs,outputs,targets,path_name,nclass=1):
    '''
    draw the result
    params:
     imgs: model input imgs(b,3,h,w) 
     outputs: (b,c,h,w)
     targets: (b,h,w)
     path_name: 'path/to/save/name.jpg' format
     nclass: class num exclude the background
    '''
    to_byte = torchvision.transforms.ConvertImageDtype(torch.uint8)
    to_float = torchvision.transforms.ConvertImageDtype(torch.float)
    def inverse_normalize(tensors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):# 避免normalize之后显示出来的图像偏黑
        for tensor in tensors:
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
        return tensors
    imgs = inverse_normalize(imgs).cpu()
    outputs = outputs.cpu()
    targets = targets.cpu()
    imgs = to_byte(imgs)
    # normalized_masks = torch.nn.functional.softmax(outputs, dim=1)# there will be a bug when using amp
    normalized_masks = outputs

    class_dim = 1
    if nclass == 1:
        colors = ['green']
    else:
        colors = None

    nclass += 1 # to include the background
    all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(nclass)[:, None, None, None]
    # The first dimension is the classes now, so we need to swap it
    all_classes_masks = all_classes_masks.swapaxes(0, 1)
    preds = [
        to_float(torchvision.utils.draw_segmentation_masks(img, masks=mask, alpha=0.7,colors=colors))
        for img, mask in zip(imgs, all_classes_masks[:,1:,:,:])
    ]

    all_classes_masks = targets == torch.arange(nclass)[:,None,None,None]
    all_classes_masks = all_classes_masks.swapaxes(0, 1)
    gts = [
        to_float(torchvision.utils.draw_segmentation_masks(img, masks=mask, alpha=0.7,colors=colors))
        for img, mask in zip(imgs, all_classes_masks[:,1:,:,:])
    ]

    grid = torchvision.utils.make_grid(preds+gts)
    torchvision.utils.save_image(grid,path_name)

if __name__ == '__main__':
    args = get_args()

    trainer = Trainer(args)
    trainer.train()