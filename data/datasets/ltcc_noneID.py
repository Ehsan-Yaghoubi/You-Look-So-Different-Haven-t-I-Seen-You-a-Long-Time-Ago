import os
import os.path as osp
from .bases import BaseImageDataset
from config import cfg

class LTCC_noneID(BaseImageDataset):
    """
    Dataset statistics:
    # identities:
    # images:  (train) +  (query) +  (gallery)
    # cameras:
    """
    dataset_dir = cfg.DATASETS.ROOT_DIR

    def __init__(self,root='', verbose=True, **kwargs):
        super(LTCC_noneID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.list_train_path = os.listdir(self.train_dir)
        self.list_gallery_path = os.listdir(self.gallery_dir)
        self.list_query_path = os.listdir(self.query_dir)
        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_dir(self.query_dir, self.list_query_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path, relabel=False)
        if verbose:
            print("=> the NoneID dataset is loaded")
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
            pid = img_info.split('_')[0] + img_info.split('_')[1]
            pid_container_orig.add(pid)
        pid_container_orig = sorted(pid_container_orig)
        pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
        for img_idx, img_info in enumerate(list_path):
            img_path = os.path.join(dir_path, img_info)
            pid = img_info.split('_')[0] + img_info.split('_')[1]
            camid = int(img_info.split('_')[2].split("c")[1])
            feat2_dir = None
            clothid = img_info.split('_')[1]
            if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
            dataset.append((img_path, pid, camid, feat2_dir, clothid))
            pid_container_after.add(pid)
        if relabel:
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
        return dataset
