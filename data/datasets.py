import random
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from PIL import Image
from torchvision import transforms

def pil_loader(path):
    with Image.open(path) as img:
        return img.convert('RGB')

class UnpairedDataset(Dataset):
    def __init__(self,root,ki67_dir:str="Ki67",he_dir:str="HE",ki67_mask_dir:str="mask",isSSIM:bool=True,H_size:int=512,W_size:int=512):
        self.root = root
        self.ki67_dir = ki67_dir
        self.he_dir = he_dir
        self.isSSIM = isSSIM
        self.ki67_mask_dir = ki67_mask_dir

        self.ki67_paths = sorted(glob(os.path.join(root,ki67_dir,"*.bmp")))
        self.he_paths = sorted(glob(os.path.join(root,he_dir,"*.jpg")))
        random.shuffle(self.he_paths)
        self.ki67_mask_paths = sorted(glob(os.path.join(root,ki67_mask_dir,"*.bmp")))

        self.trans = transforms.Compose([
            transforms.Resize((H_size,W_size)),
            transforms.ToTensor()
        ])



    def __len__(self):
        return max(len(self.ki67_paths),len(self.he_paths))

    def __getitem__(self,index):
        ki67_path = self.ki67_paths[index%len(self.ki67_paths)]
        he_path = self.he_paths[index%len(self.he_paths)]
        ki67_mask_path = self.ki67_mask_paths[index%len(self.ki67_paths)]

        ki67_data = self.trans(pil_loader(ki67_path))
        he_data = self.trans(pil_loader(he_path))
        print(ki67_path)
        print(he_path)

        sample = {
            "ki67_data":ki67_data,
            "he_data":he_data,
            "ki67_mask_data":None
        }

        if self.isSSIM:
            mask = Image.open(ki67_mask_path).convert("L")
            mask = self.trans(mask)
            sample["ki67_mask_data"] = (mask>0.5).float()

        return sample


if __name__ == "__main__":
    root_dir = r"F:\科研\细胞分割\DDPM\data"
    dataset = UnpairedDataset(
        root=root_dir,
        ki67_dir="Ki67",
        he_dir="HE",
        ki67_mask_dir="mask",
        isSSIM=True,
        H_size=256,
        W_size=256
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print("✔ Dataset 加载成功")
    print("Ki67 图像:", batch["ki67_data"].shape)  # [B, 3, H, W]
    print("HE 图像:", batch["he_data"].shape)  # [B, 3, H, W]
    if batch["ki67_mask_data"][0] is not None:
        print("Ki67 掩码:", batch["ki67_mask_data"].shape)  # [B, 1, H, W]





