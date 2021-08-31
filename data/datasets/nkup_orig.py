import os
import os.path as osp
from config import cfg
from .bases import BaseImageDataset

class NKUP_Orig(BaseImageDataset):
    """
    For knowing how to read annotations please go to this link
    https://github.com/nkicsl/NKUP-dataset/issues/2

    Dataset statistics:
      subset   | # ids | # images | # cameras
      ----------------------------------------
      train    |     |      |
      query    |     |      |
      gallery  |     |      |
    """
    dataset_dir = cfg.DATASETS.ROOT_DIR

    def __init__(self,root='',is_train='', verbose=True, **kwargs):
        super(NKUP_Orig, self).__init__()
        self.is_train = is_train
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.list_train_path = os.listdir(self.train_dir)
        self.list_gallery_path = os.listdir(self.gallery_dir)
        self.list_query_path = os.listdir(self.query_dir)
        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_dir(self.query_dir, self.list_query_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path, relabel=False)
        if verbose:
            print("=> the original dataset is loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir): raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir): raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir): raise RuntimeError("'{}' is not available".format(self.query_dir))

    def _process_dir(self, dir_path, list_path, relabel=False):
        dataset = []
        pid_container_orig = set()
        pid_container_after = set()
        for img_idx, img_info in enumerate(list_path):
            pid = img_info.split('_')[0]
            pid_container_orig.add(pid)
        pid_container_orig = sorted(pid_container_orig)
        pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
        print("*" * 50)
        print(dir_path)
        print(pid2label)
        print("*" * 50)
        for img_idx, img_info in enumerate(list_path):
            img_path = os.path.join(dir_path, img_info)
            pid = img_info.split('_')[0]
            camid = int(img_info.split('_')[2].split("S")[0].split("C")[1])
            clothid = img_info.split('_')[1].split("D")[1]
            feat2_dir_1 = None
            #feat2_dir_2 = None
            if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
            if cfg.DATASETS.feat2_DIR is not None:
                feat2_dir_1 = os.path.join(cfg.DATASETS.feat2_DIR, "_1_{}.npy".format(img_info))
                #feat2_dir_2 = os.path.join(cfg.DATASETS.feat2_DIR, "_1_{}.npy".format(img_info))
            dataset.append((img_path, pid, camid, feat2_dir_1, clothid))
            #dataset.append((img_path, pid, camid, feat2_dir_2, clothid))
            pid_container_after.add(pid)
        if relabel:
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
        return dataset
