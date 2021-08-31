#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import re

import os
import os.path as osp
from config import cfg
from .bases import BaseImageDataset


class CELEBREID_Orig(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    #TRAIN_MODE = "LTCC_with_origIDs" #  "LTCC_orig_origIDs_withclothid" | "LTCC_orig_origIDs" | "LTCC_orig_new_IDs" | "LTCC_orig" | "LTCC_nonID"
    #dataset_dir = '/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_orig'
    #feat2_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_nonID/scratched/NoneID_features_N1"
    # feat2_dir = None
    dataset_dir = cfg.DATASETS.ROOT_DIR
    # TRAIN_MODE = cfg.TRAIN_MODE
    # feat2_dir = cfg.Save_nonIDs_DIR

    def __init__(self,root='',is_train='', verbose=True, **kwargs):
        super(CELEBREID_Orig, self).__init__()
        self.is_train = is_train
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.query_dir = osp.join(self.dataset_dir, 'query')

        self.list_train_path = os.listdir(self.train_dir)
        self.list_gallery_path = os.listdir(self.gallery_dir)
        self.list_query_path = os.listdir(self.query_dir)

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_dir(self.query_dir, self.list_query_path)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path)
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
            if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
            camid = img_info.split('_')[0] + img_info.split('_')[1] + img_info.split('_')[2]
            clothid = img_info.split('_')[1]
            if cfg.Save_nonIDs_DIR is not None:  # if you want to run STE_CNN, make cfg.Save_nonIDs_DIR=None
                feat2_dir = os.path.join(cfg.Save_nonIDs_DIR, "{}.npy".format(img_info))
            else: feat2_dir = None

            dataset.append((img_path, pid, camid, feat2_dir, clothid))
            pid_container_after.add(pid)
        if relabel:
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
        return dataset
        #
        #     elif cfg.TRAIN_MODE == "assign_new_IDs":
        #         for img_idx, img_info in enumerate(list_path):
        #             pid = img_info.split('_')[0] + img_info.split('_')[1]
        #             pid_container_orig.add(pid)
        #         pid_container_orig = sorted(pid_container_orig)
        #         pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
        #         ### important notice: When you want to relable the data with IDs from 0 to N, you must check that
        #         ###   the same person in the gallery and query should receive an identical ID label
        #         print("*" * 50)
        #         print(dir_path)
        #         print(pid2label)
        #         print("*" * 50)
        #         for img_idx, img_info in enumerate(list_path):
        #             img_path = os.path.join(dir_path, img_info)
        #             pid = img_info.split('_')[0] + img_info.split('_')[1]
        #             person_id = pid2label[pid]  # train ids must be relabelled from zero
        #             camid = img_info.split('_')[0] + img_info.split('_')[1] + img_info.split('_')[2]
        #             feat2_dir = None
        #             clothid = img_info.split('_')[1]
        #             if cfg.Save_nonIDs_DIR is not None:  # if you want to run STE_CNN, make cfg.Save_nonIDs_DIR=None
        #                 feat2_dir = os.path.join(cfg.Save_nonIDs_DIR, "{}.npy".format(img_info))
        #             dataset.append((img_path, person_id, camid, feat2_dir, clothid))
        #             pid_container_after.add(person_id)
        #
        #         # check if pid starts from 0 and increments with 1
        #         for idx, pid in enumerate(pid_container_after):
        #             assert idx == pid, "See code comment for explanation"
        #         return dataset
        #
        #     else:
        #
        #         raise ValueError("set the TRAIN_MODE parameter in config file")
        #
        # elif not self.is_train:
        #     if cfg.TEST.MODE == "original_IDs":
        #         for img_idx, img_info in enumerate(list_path):
        #             pid = img_info.split('_')[0]
        #             pid_container_orig.add(pid)
        #         pid_container_orig = sorted(pid_container_orig)
        #         pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
        #         ### important notice: When you want to relable the data with IDs from 0 to N, you must check that
        #         ###   the same person in the gallery and query should receive an identical ID label
        #         print("*" * 50)
        #         print(dir_path)
        #         print(pid2label)
        #         print("*" * 50)
        #         for img_idx, img_info in enumerate(list_path):
        #             img_path = os.path.join(dir_path, img_info)
        #             pid = img_info.split('_')[0]
        #             person_id = pid2label[pid]  # train ids must be relabelled from zero
        #             camid = img_info.split('_')[0] + img_info.split('_')[1] + img_info.split('_')[2]
        #             clothid = img_info.split('_')[1]
        #             feat2_dir = None
        #             if cfg.Save_nonIDs_DIR is not None:  # if you want to run STE_CNN, make cfg.Save_nonIDs_DIR=None
        #                 feat2_dir = os.path.join(cfg.Save_nonIDs_DIR, "{}.npy".format(img_info))
        #             dataset.append((img_path, person_id, camid, feat2_dir, clothid))
        #             pid_container_after.add(person_id)
        #
        #         # check if pid starts from 0 and increments with 1
        #         for idx, pid in enumerate(pid_container_after):
        #             assert idx == pid, "See code comment for explanation"
        #         return dataset
        #
        #     elif cfg.TEST.MODE == "new_IDs":
        #         for img_idx, img_info in enumerate(list_path):
        #             pid = img_info.split('_')[0] + img_info.split('_')[1]
        #             pid_container_orig.add(pid)
        #         pid_container_orig = sorted(pid_container_orig)
        #         pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
        #         ### important notice: When you want to relable the data with IDs from 0 to N, you must check that
        #         ###   the same person in the gallery and query should receive an identical ID label
        #         print("*" * 50)
        #         print(dir_path)
        #         print(pid2label)
        #         print("*" * 50)
        #         for img_idx, img_info in enumerate(list_path):
        #             img_path = os.path.join(dir_path, img_info)
        #             pid = img_info.split('_')[0] + img_info.split('_')[1]
        #             person_id = pid2label[pid]  # train ids must be relabelled from zero
        #             camid = img_info.split('_')[0] + img_info.split('_')[1] + img_info.split('_')[2]
        #             feat2_dir = None
        #             clothid = img_info.split('_')[1]
        #             if cfg.Save_nonIDs_DIR is not None:  # if you want to run STE_CNN, make cfg.Save_nonIDs_DIR=None
        #                 feat2_dir = os.path.join(cfg.Save_nonIDs_DIR, "{}.npy".format(img_info))
        #             dataset.append((img_path, person_id, camid, feat2_dir, clothid))
        #             pid_container_after.add(person_id)
        #
        #         # check if pid starts from 0 and increments with 1
        #         for idx, pid in enumerate(pid_container_after):
        #             assert idx == pid, "See code comment for explanation"
        #         return dataset
        #
        #     else:
        #         raise ValueError("set the TEST.MODE parameter in config file")
        #
        # else:
        #     raise ValueError("Set is_train as True or False")
