
import argparse
import torch
from torch.utils.data import DataLoader

from utils.dataset import HorseDataset,BaseTransform
from utils.metric import SegmentationMetric
from model import DenseASPP,FastSCNN


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path',type=str,default='asset/model/DenseASPP_horse_best.pth',
                        help='model path')
    parser.add_argument('--model',type=str,default='DenseASPP',choices=['DenseASPP','FastSCNN'],
                        help='which model to eval')
    parser.add_argument('--batch_size',type=int,default=8,
                        help='batch size')
    parser.add_argument('--data_dir',type=str,default='data/horse_data',
                        help='path to data')
    args = parser.parse_args()
    return args

args = get_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset and dataloader
transforms = BaseTransform(resize_size=[512,512])
dataset = HorseDataset(args.data_dir,transforms)
data_num = len(dataset)
train_num = int(data_num*0.85)
train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_num,data_num-train_num])
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,pin_memory=True)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=2,pin_memory=True)

nclass = 1

# model
if args.model.lower() == 'denseaspp':
    from torchvision.models import DenseNet121_Weights
    weights = DenseNet121_Weights.DEFAULT
    model = DenseASPP(nclass=nclass,pretrained_weights=weights)
elif args.model.lower() == 'fastscnn':
    model = FastSCNN(nclass)

model.load_state_dict(torch.load(args.model_path))

# metric
metric = SegmentationMetric(nclass = nclass)

#---------
# start eval
#---------
print('eval %s on %s dataset:'%(args.model,'horse'))
metric.reset()
model.eval()
model.to(device)
for data in test_dataloader:
    imgs,targets = data
    imgs = imgs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        outputs = model(imgs)
    metric.update(outputs,targets)

pixAcc,mIoU,bIoU = metric.get()
print('result: pixel accuracy: %f\tmIoU: %f\tbIoU: %f'%(pixAcc,mIoU,bIoU))
