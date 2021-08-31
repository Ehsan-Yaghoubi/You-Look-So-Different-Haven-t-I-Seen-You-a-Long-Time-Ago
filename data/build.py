from torch.utils.data import DataLoader
from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid
from .transforms import build_transforms

def make_data_loader(cfg, is_train):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Initialize the dataset/datasets
    if cfg.DATASETS.multiple:
        # TODO: add multi dataset to train and test
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, is_train=is_train)
    else:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, is_train=is_train)
    # ID sampling. See more codes at https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/data/sampler.html
    if cfg.DATALOADER.IDsampler == True:
        sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,sampler=sampler, shuffle=shuffle,num_workers=num_workers, collate_fn=train_collate_fn) # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,collate_fn=val_collate_fn)

    return train_loader, val_loader, len(dataset.query), num_classes
