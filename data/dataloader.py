from datasets import UnpairedDataset
from torch.utils.data import DataLoader
import yaml
import torch
def build_train_dataloader(cfg):
    """
    cfg 应该包含 data 相关字段：
      - root, ki67_dir, he_dir, ki67_mask_dir
      - batch, workers, H_size, W_size
    """

    c = cfg['data']
    dataset = UnpairedDataset(
        root = c['root'],
        ki67_dir=c.get('ki67_dir', 'Ki67'),
        he_dir=c.get('he_dir', 'HE'),
        ki67_mask_dir=c.get('ki67_mask_dir', 'mask'),
        isSSIM=c.get('use_mask', True),
        H_size=c.get('H_size', 256),
        W_size=c.get('W_size', 256)
    )


    return DataLoader(
        dataset,
        batch_size=c.get('batch',4),
        shuffle=False,
        num_workers=c.get('workers',0),
        pin_memory=True
    )

def build_infer_dataloader(cfg):
    """
    构建用于推理的 DataLoader。
    通常不打乱数据，且可能不需要 mask（如未启用结构监督）。
    """
    c = cfg['data']
    dataset = UnpairedDataset(
        root=c['root'],
        ki67_dir=c.get('ki67_dir', 'Ki67'),
        he_dir=c.get('he_dir', 'HE'),  # 如果需要对比生成结果和真实HE
        ki67_mask_dir=c.get('ki67_mask_dir', 'mask'),
        isSSIM=c.get('use_mask', False),  # 推理时一般不需要mask
        H_size=c.get('H_size', 256),
        W_size=c.get('W_size', 256)
    )

    return DataLoader(
        dataset,
        batch_size=c.get('batch', 4),
        shuffle=False,
        num_workers=c.get('workers', 0),
        pin_memory=True
    )


def test_dataloader(dataloader, visualize: bool = True):
    print("🚀 正在测试 DataLoader...")
    batch = next(iter(dataloader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("../configs/train_config.yaml"))
    train_dataloader = build_train_dataloader(cfg)
    print(cfg['data'].get('batch'))
    test_dataloader(train_dataloader)