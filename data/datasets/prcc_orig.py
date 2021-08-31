#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import re

import os
import os.path as osp
from config import cfg
from .bases import BaseImageDataset
import random

class PRCC_Orig(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    query_camera = cfg.DATASETS.CAMERA
    dataset_dir = cfg.DATASETS.ROOT_DIR
    def __init__(self,root='',is_train='', verbose=True, **kwargs):
        super(PRCC_Orig, self).__init__()
        self.query_camera = cfg.DATASETS.CAMERA
        self.is_train = is_train
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        #  we should randomly select one image of each person in the A folder, as gallery
        #  source: https://ieeexplore.ieee.org/document/8936426
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'A')
        # Folders B and C are considered as prob. In folder B, persons have same clothes with gallery (i.e., folder A),
        # but in C persons have different clothes from the gallery.
        self.query_dir = osp.join(self.dataset_dir, 'test', self.query_camera) # 'B' or 'C' for same clothes and different clothes, respectively

        self.list_train_path = os.listdir(self.train_dir)
        self.list_gallery_path = os.listdir(self.gallery_dir)
        self.list_query_path = os.listdir(self.query_dir)

        self._check_before_run()
        train = self.train_process_dir(self.train_dir, relabel=True)
        query = self.query_process_dir(self.query_dir, self.list_query_path)
        gallery = self.gallery_process_dir(self.gallery_dir, self.list_gallery_path)
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
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        # if cfg.TRAIN_MODE == "orig_IDs":
        #     if not osp.exists(cfg.MODEL.NETWORK == "LTE_CNN"):
        #         # check to see if feat2 is available (feat2 is needed for our loss function)
        #         raise RuntimeError("'{}' is not available".format(cfg.Save_nonIDs_DIR))

    def train_process_dir(self, dir_path, relabel=False):
        dataset = []
        pid_container_after = set()
        person_ids = os.listdir(dir_path)
        pid_container_orig = sorted(set(person_ids))
        # give ids from zero
        pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
        print("*" * 50)
        print(dir_path)
        print(pid2label)
        print("*" * 50)

        # prepare the dataset
        for indx, folder_id in enumerate(person_ids):
            imgs_list = os.listdir(os.path.join(dir_path, folder_id))
            for img_idx, img_info in enumerate(imgs_list):
                img_path = os.path.join(dir_path, folder_id, img_info)
                camera = img_info.split('_')[0]
                if camera == "A":  ## camid: A=1, B=2, C=3  ## clothid: A=4, B=4, C=5
                    clothid = 4
                    camid = 1
                elif camera == "B":
                    clothid = 4
                    camid = 2
                elif camera == "C":
                    clothid = 5
                    camid = 3
                else:
                    raise ValueError("check the codes for a semantic error!")
                if relabel: pid = pid2label[folder_id]
                else: pid = folder_id

                if cfg.DATASETS.feat2_DIR is not None:
                    feat2_dir = os.path.join(cfg.DATASETS.feat2_DIR, '{}.npy'.format(img_info))
                else:
                    feat2_dir = None
                dataset.append((img_path, pid, camid, feat2_dir, clothid))
                pid_container_after.add(pid)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container_after):
            assert idx == pid, "See code comment for explanation"
        return dataset

    def gallery_process_dir(self, dir_path, list_path):
        # gallery is taken from folder A. So, clothid=4 and camid=1
        dataset = []
        person_ids = os.listdir(dir_path)

        for indx, id_folder in enumerate(person_ids):
            imgs_list = os.listdir(os.path.join(dir_path, id_folder))
            random_img_of_this_id = random.choice(imgs_list)
            img_path = os.path.join(dir_path, id_folder, random_img_of_this_id)
            pid = id_folder
            camid = 1  # A = 1, B=2, C=3
            feat2_dir = None
            clothid = 4  # A = 4, B=4, C=5
            dataset.append((img_path, pid, camid, feat2_dir, clothid))

        return dataset

    def query_process_dir(self, dir_path, list_path):
        # query is taken from folder B (so, clothid=4 and camid=2) or it is taken folder C (so, clothid=5 and camid=3)
        folder = dir_path.split("/")[-1]
        dataset = []
        person_ids = os.listdir(dir_path)

        # prepare the dataset
        for indx, id_folder in enumerate(person_ids):
            imgs_list = os.listdir(os.path.join(dir_path, id_folder))
            for img_idx, img_info in enumerate(imgs_list):
                img_path = os.path.join(dir_path, id_folder, img_info)
                if folder == "B":
                    camid = 2
                    clothid = 4
                elif folder == "C":
                    camid = 3
                    clothid = 5
                else:
                    raise ValueError("check the codes for a semantic error!")
                feat2_dir = None
                pid = id_folder
                dataset.append((img_path, pid, camid, feat2_dir, clothid))
        return dataset


