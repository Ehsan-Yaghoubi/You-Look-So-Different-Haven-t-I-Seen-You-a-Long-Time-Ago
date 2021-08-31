import os
import sys
import torch
import numpy as np
sys.path.append('.')
from config import cfg
from data.transforms import build_transforms
from modeling.baseline import Baseline
from PIL import Image
import argparse


def load_image(image_name):
    image = Image.open(image_name)
    val_transforms = build_transforms(cfg, is_train=False)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    return image

def load_trained_model(cfg, checkpoint_path):
    num_classes = 10

    #_model = Baseline(num_classes, 1, '/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/_01_preprocessing_step/resnet50-19c8e357.pth', 'bnneck', 'after', 'resnet50', 'imagenet')
    _model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    _model.load_param(checkpoint_path)
    _model.eval()
    return _model

def save_inferenced_features(cfg, save_feat_dir, checkpoint_path):
    os.makedirs(save_feat_dir, exist_ok=True)
    main_dir = cfg.DATASETS.ROOT_DIR
    # prepare the model
    model = load_trained_model(cfg, checkpoint_path)
    print("checkpoint was loaded successfully. Next, images are fed to the model one by one for feature extraction.")
    folders = os.listdir(main_dir)

    for index, name in enumerate(folders):
        if os.path.isdir(os.path.join(main_dir, name)):
            for fol, folder_name in enumerate(folders):
                sub_names = os.listdir(os.path.join(main_dir,folder_name))
                print("Done Parent Folders: \t{}/{}".format(fol, len(folders)))

                for i, sub__name in enumerate(sub_names):

                    if os.path.isdir(os.path.join(main_dir, folder_name, sub__name)):
                        pid_imgs = os.listdir(os.path.join(main_dir, folder_name, sub__name))
                        print("Done child1 Folders: \t{}/{}".format(i, len(sub_names)))
                        for ii, sub_sub_name in enumerate(pid_imgs):

                            if os.path.isdir(os.path.join(main_dir, folder_name, sub__name, sub_sub_name)):
                                sub_sub_files = os.listdir(os.path.join(main_dir, folder_name, sub__name, sub_sub_name))

                                for iii, sub_sub_file_name in enumerate(sub_sub_files):

                                    image = load_image(os.path.join(main_dir, folder_name, sub__name, sub_sub_name, sub_sub_file_name))
                                    with torch.no_grad():
                                        features = model(image)
                                        feature_arry = features.numpy()
                                        file = os.path.join(save_feat_dir, "{}.npy".format(sub_sub_file_name))
                                        np.save(file=file, arr=feature_arry)
                                    print("sub_sub_sub: feature extraction: \t{}/{}".format(iii, len(sub_sub_files)))
                            else:
                                image = load_image(os.path.join(main_dir, folder_name, sub__name, sub_sub_name))
                                with torch.no_grad():
                                    features = model(image)
                                    feature_arry = features.numpy()
                                    file = os.path.join(save_feat_dir, "{}.npy".format(sub_sub_name))
                                    np.save(file=file, arr=feature_arry)
                                print("sub_sub: feature extraction: \t{}/{}".format(ii, len(pid_imgs)))

                    else:
                        image = load_image(os.path.join(main_dir, folder_name, sub__name))
                        with torch.no_grad():
                            features = model(image)
                            feature_arry = features.numpy()
                            file = os.path.join(save_feat_dir, "{}.npy".format(sub__name))
                            np.save(file=file, arr=feature_arry)
                        print("sub: feature extraction: \t{}/{}".format(i, len(sub_names)))

        else:
            image = load_image(os.path.join(main_dir, name))
            with torch.no_grad():
                features = model(image)
                feature_arry = features.numpy()
                file = os.path.join(save_feat_dir, "{}.npy".format(name))
                np.save(file=file, arr=feature_arry)

            print("feature extraction: \t{}/{}".format(index, len(folders)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="../configs/strong_baseline.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_feat_dir_STB = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/YLD_YouLookDifferent/OUTPUT/tsne/strongbaseline_long_term_representations_20ids_testdata"
    checkpoint_path_STB = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/YLD_YouLookDifferent/OUTPUT/strong_baseline/train_2021_Mar_08_12_31_22/resnet50_model_50.pth"

    save_feat_dir_BSTF = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/YLD_YouLookDifferent/OUTPUT/tsne/BSTF_long_term_representations_20ids_testdata"
    checkpoint_path_BSTF = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/YLD_YouLookDifferent/OUTPUT/Network2/paper_256to128_senet154/senet154_model_235.pth"


    save_feat_dir = save_feat_dir_STB  # change .yml file also
    checkpoint_path = checkpoint_path_STB # change .yml file also

    save_inferenced_features(cfg, save_feat_dir, checkpoint_path)
