import argparse
import os
import sys
from os import mkdir
import torch
from torch.backends import cudnn
from datetime import datetime
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger

def main(logging=True):
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument("--config_file", default="../configs/LTE_CNN.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if logging:
        time = datetime.now()
        time = time.strftime("test_%Y_%m_%d_%H_%M_%S")

        output_dir = cfg.OUTPUT_DIR
        output_dir = os.path.join(output_dir, time)
        if output_dir and not os.path.exists(output_dir):
            mkdir(output_dir)

        logger = setup_logger("TEST clothing change re-id", output_dir, 0, time)
        logger.info("Using {} GPUS".format(num_gpus))
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
    else:
        logger=None

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg, is_train=False)
    model = build_model(cfg, num_classes)
    print("cfg.TEST.WEIGHT: ", cfg.TEST.WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)

    ## replace 'model.load_param(cfg.TEST.WEIGHT)' with the following code
    # checkpoint = torch.load(cfg.TEST.WEIGHT)
    # model.load_state_dict(checkpoint['model'])
    # model.eval()
    # print("pretrained model is loaded from: ",cfg.TEST.WEIGHT)

    _CC_cmc, _CC_mAP, _SS_cmc, _SS_mAP = inference(cfg, model, val_loader, num_query)
    return _CC_cmc, _CC_mAP, _SS_cmc, _SS_mAP, logger

def test_after_train(cfg, path_to_weights, output_dir, time, args):

    logger = setup_logger("TEST clothing change re-id", output_dir, 0, time)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg, is_train=False)
    model = build_model(cfg, num_classes)
    model.load_param(path_to_weights)

    _CC_cmc, _CC_mAP, _SS_cmc, _SS_mAP = inference(cfg, model, val_loader, num_query)

    return _CC_cmc, _CC_mAP, _SS_cmc, _SS_mAP

if __name__ == '__main__':

    one_time_test = True

    if one_time_test:

        main()

    else:
        import numpy as np
        Total_CC_cmc_ = []
        Total_CC_mAP_ = []
        Total_SS_cmc_ = []
        Total_SS_mAP_ = []
        var_CC_cmc = 0
        var_CC_mAP = 0
        var_SS_cmc = 0
        var_SS_mAP = 0
        CC_cmc = []
        CC_mAP = 0
        SS_cmc = []
        SS_mAP = 0

        for i in range (0, 10, 1):
            print ("Test {}. Repeating test for 10 times and caclulating the avg.".format(i))
            CC_cmc_, CC_mAP_, SS_cmc_, SS_mAP_, logger = main(logging = False)
            if CC_cmc_ is not None: Total_CC_cmc_.append(100*np.array(CC_cmc_))
            if CC_mAP_ is not None: Total_CC_mAP_.append(100*np.array(CC_mAP_))
            if SS_cmc_ is not None: Total_SS_cmc_.append(100*np.array(SS_cmc_))
            if SS_mAP_ is not None: Total_SS_mAP_.append(100*np.array(SS_mAP_))

        if len(Total_CC_cmc_)!=0: CC_cmc = sum(Total_CC_cmc_) / len(Total_CC_cmc_)
        if len(Total_CC_mAP_)!=0: CC_mAP = sum(Total_CC_mAP_) / len(Total_CC_mAP_)
        if len(Total_SS_cmc_)!=0: SS_cmc = sum(Total_SS_cmc_) / len(Total_SS_cmc_)
        if len(Total_SS_mAP_)!=0: SS_mAP = sum(Total_SS_mAP_) / len(Total_SS_mAP_)

        if len(Total_CC_cmc_)!=0: var_CC_cmc = sum((x-CC_cmc)**2 for x in Total_CC_cmc_) / (len(Total_CC_cmc_)-1)
        if len(Total_CC_mAP_)!=0: var_CC_mAP = sum((x-CC_mAP)**2 for x in Total_CC_mAP_) / (len(Total_CC_mAP_)-1)
        if len(Total_SS_cmc_)!=0: var_SS_cmc = sum((x-SS_cmc)**2 for x in Total_SS_cmc_) / (len(Total_SS_cmc_)-1)
        if len(Total_SS_mAP_)!=0: var_SS_mAP = sum((x-SS_mAP)**2 for x in Total_SS_mAP_) / (len(Total_SS_mAP_)-1)

        if len(Total_SS_mAP_) != 0:
            print("Test: Standard reid Setting.")
            print("mAP: {}\t variance:{}".format(SS_mAP, var_SS_mAP))
            for r in [1, 5, 10, 20, 50]:
                print("CMC curve, Rank-{:<3}:{}:\t variance:{}".format(r, SS_cmc[r - 1], var_SS_cmc[r - 1]))

        if len(Total_CC_mAP_) != 0:
            print("Test: CC reid Setting.")
            print("mAP: {}\t variance:{}".format(CC_mAP, var_CC_mAP))
            print()
            for r in [1, 5, 10, 20, 50]:
                print("CMC curve, Rank-{:<3}:{}\t variance:{}".format(r, CC_cmc[r - 1], var_CC_cmc[r - 1]))
