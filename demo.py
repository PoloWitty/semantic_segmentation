
import argparse
import torch
from torch.utils.data import DataLoader

from utils.dataset import HorseDataset,BaseTransform
from model import DenseASPP,FastSCNN
from train import save_result

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path',type=str,default='asset/model/DenseASPP_horse_best.pth',
                        help='model path')
    parser.add_argument('--model',type=str,default='DenseASPP',choices=['DenseASPP','FastSCNN'],
                        help='which model to eval')
    args = parser.parse_args()
    return args

args = get_args()

# dataset and dataloader
transforms = BaseTransform(resize_size=[512,512])
dataset = HorseDataset('asset/demo_data',transforms)
dataloader = DataLoader(dataset,batch_size=8,num_workers=2,pin_memory=True)

nclass = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model
if args.model.lower() == 'denseaspp':
    from torchvision.models import DenseNet121_Weights
    weights = DenseNet121_Weights.DEFAULT
    model = DenseASPP(nclass=nclass,pretrained_weights=weights)
elif args.model.lower() == 'fastscnn':
    model = FastSCNN(nclass)
model.load_state_dict(torch.load(args.model_path))
model.to(device)

# data
data = next(iter(dataloader))
imgs,targets = data
imgs = imgs.to(device)
targets = targets.to(device)

# infer and save the result
model.eval()
outputs = model(imgs)
save_result(imgs,outputs,targets,'asset/%s_demo_result.jpg'%args.model,nclass)
print('save result to "asset/demo_result"')
print('the former is predictions, the latter is gt')