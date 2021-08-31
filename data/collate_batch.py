import torch

def train_collate_fn(batch):
    imgs, pids, _, _, feat2, clothid = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, feat2, clothid


def val_collate_fn(batch):
    imgs, pids, camids, _, feat2, clothid = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, clothid
