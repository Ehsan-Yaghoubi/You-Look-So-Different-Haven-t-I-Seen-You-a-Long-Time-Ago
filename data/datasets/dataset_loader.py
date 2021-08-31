import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    img = None
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset_"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, feat2_path, clothid = self.dataset[index]
        img = read_image(img_path)
        if feat2_path is not None:
            try:
                feat2 = np.load(feat2_path)
                #print("feat2 is loaded from {}".format(feat2_path))
            except FileNotFoundError:
                #print("features for this image is not found: {}".format(feat2_path))
                feat2 = np.ones((1,2048), dtype=np.float32)
        else: #feat2 = None
            feat2 = np.ones((1, 2048), dtype=np.float32)

        if self.transform is not None: img = self.transform(img)

        return img, pid, camid, img_path, feat2, int(clothid)



