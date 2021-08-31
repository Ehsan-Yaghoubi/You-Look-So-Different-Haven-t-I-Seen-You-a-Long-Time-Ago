import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from datetime import datetime
from tools import net_test, test_on_one_img
from utils.logger import setup_logger
#import hiddenlayer as hl

############### feat2 is loaded from

def train(cfg, time):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg, is_train=True)
    # prepare model
    # mymodel = build_model(cfg, num_classes)
    # print(">> model is built")
    # # save the model architecture in a .pdf file. More codes at https://github.com/waleedka/hiddenlayer/blob/master/demos/pytorch_graph.ipynb
    # hl_graph = hl.build_graph(mymodel, (torch.zeros([1, 3, 128, 128])))
    # hl_graph.save("graph_model")

    model = build_model(cfg, num_classes)
    print(">> model is built")

    arguments = {}
    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is: **{}**'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        optimizer = make_optimizer(cfg, model)
        loss_func = make_loss(cfg, num_classes)
        # Add for using (your) trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))
        # To use yourself trained model, 'scheduler' and 'start_epoch' should be modified
        do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,loss_func, num_query,start_epoch,time)

    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        arguments = {}
        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            center_criterion.load_state_dict(torch.load(path_to_center_param))
            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))
        do_train_with_center(cfg,model,center_criterion,train_loader,val_loader,optimizer,optimizer_center,scheduler, loss_func,num_query,start_epoch)

    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="../configs/LTE_CNN.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    time_tmp = datetime.now()
    time = time_tmp.strftime("train_%Y_%h_%d_%H_%M_%S")
    time_test = time_tmp.strftime("test_%Y_%h_%d_%H_%M_%S")

    output_dir = cfg.OUTPUT_DIR
    output_dir = os.path.join(output_dir, time)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("clothing change re-id", output_dir, 0, time)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True

    train(cfg, time)

    if cfg.MODEL.NETWORK == "simple_baseline" or cfg.MODEL.NETWORK == "strong_baseline":
        import numpy as np
        print(">> Baseline is Trained. Check the log files and model checkpoints at {}".format(cfg.OUTPUT_DIR))
        print(">> Baseline will be tested for 2 times and the mean and variance will be reported")
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

        for i in range (0, 2, 1):
            print ("Test {}. Repeating test for 10 times and calculating the avg.".format(i))
            CC_cmc_, CC_mAP_, SS_cmc_, SS_mAP_, logger = net_test.main(logging = False)
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


    elif cfg.MODEL.NETWORK == "STE_CNN":
        print(">> STE_CNN was Trained. Now, dataset features will be extracted and saved on the disk as .npy arrays at {}".format(cfg.OUTPUT_DIR))
        checkpoint_path = os.path.join(output_dir, "{}_model_{}.pth".format(cfg.MODEL.NAME, cfg.SOLVER.MAX_EPOCHS))
        save_feat_dir = cfg.Save_nonIDs_DIR
        test_on_one_img.save_inferenced_features(cfg, save_feat_dir, checkpoint_path)

    elif cfg.MODEL.NETWORK == "LTE_CNN":
        print(">> LTE_CNN was Trained. Check the log file and model checkpoints at {}".format(cfg.OUTPUT_DIR))
        print(">> Test phase based on the last checkpoint: ")
        path_to_weights = os.path.join(output_dir, "{}_model_{}.pth".format(cfg.MODEL.NAME, cfg.SOLVER.MAX_EPOCHS))
        net_test.test_after_train(cfg, path_to_weights, output_dir, time_test, args)

    else:
        raise ValueError("choose the \'cfg.MODEL.NETWORK\' between \'simple_baseline\', \'strong_baseline\', \'STE_CNN\', \'LTE_CNN\'")


if __name__ == '__main__':
    main()

