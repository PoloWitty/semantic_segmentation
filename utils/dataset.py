import torch
import glob
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torch import nn,Tensor
from typing import Optional, Tuple
import random

from torchvision.io import read_image

class BaseTransform(nn.Module):
    def __init__(
        self,
        *,
        resize_size: Optional[int],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.resize_size = resize_size if resize_size is not None else None
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor, mask: Tensor) -> Tensor:
        '''
        output shape: 
         img: (b,3,resize_size[0],resize_size[1])
         mask: (b,resize_size[0],resize_size[1])
        '''
        if isinstance(self.resize_size, list):
            img = F.resize(img, self.resize_size, interpolation=self.interpolation)
            mask = F.resize(mask, self.resize_size, interpolation=self.interpolation)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        if not isinstance(mask, Tensor):
            mask = F.pil_to_tensor(mask)
        img = F.convert_image_dtype(img, torch.float)
        # mask = F.convert_image_dtype(mask,torch.long) # this will scale the num which is not expected
        mask = mask.long()
        mask.squeeze_() # (h,w)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img , mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image and mask``torch.Tensor`` objects. "
            f"The images and masks are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the images are first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and "
            f"``std={self.std}``."
        )

class AugTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,imgs,masks):
        '''
        params:
         imgs: (b,3,h,w)
         masks: (b,h,w)
        '''
        img_aug = [imgs]
        mask_aug = [masks]
        if random.random() > 0.5:
            img_aug.append(F.hflip(imgs))
            mask_aug.append(F.hflip(masks))
        else:
            angle = random.choice([-30, -15, 0, 15, 30])
            img_aug.append(F.rotate(imgs, angle))
            mask_aug.append(F.rotate(masks,angle))
        return torch.cat(img_aug,dim=0),torch.cat(mask_aug,dim=0)


class HorseDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,transforms):
        self.transforms = transforms
        self.img_files = sorted(glob.glob(data_path + '/horse' + "/*.*"))
        self.mask_files = sorted(glob.glob(data_path + '/mask' +'/*.*'))

    def __getitem__(self,index):
        img = read_image(self.img_files[index % len(self.img_files)])
        mask = read_image(self.mask_files[index % len(self.mask_files)])
        img , mask = self.transforms(img,mask)
        return img,mask
    
    def __len__(self):
        return len(self.img_files)

class AccHorseDataset(torch.utils.data.Dataset):
    def __init__(self,data_path) -> None:
        self.files = sorted(glob.glob(data_path+"/*.*"))
    
    def __getitem__(self,index):
        return torch.load(self.files[index % len(self.files)])

    def __len__(self):
        return len(self.files)

if __name__=='__main__':
    from torch.utils.data import DataLoader
    import time

    transforms = BaseTransform(resize_size = [512,512])
    dataset = HorseDataset('data/horse_data/',transforms)

    dataloader = DataLoader(dataset,batch_size=8)
    print(len(dataset))
    start = time.time()
    for i,data in enumerate(dataloader):
        img, mask = data
        # print(img.shape)
        # print(mask.shape)
        end = time.time()
        # break
        torch.save((img[0],mask[0]),'data/accHorse/%i.pt'%i)
    print('before acc: %f'%(end-start))


    dataset = AccHorseDataset('data/accHorse')
    dataloader = DataLoader(dataset,batch_size=8)
    aug_transform = AugTransform()
    start = time.time()
    for data in dataloader:
        img,mask = data
        img,mask = aug_transform(img,mask)
        # print(img.shape)
        # print(mask.shape)
        end = time.time()
        break
    print('after acc: %f'%(end-start))
