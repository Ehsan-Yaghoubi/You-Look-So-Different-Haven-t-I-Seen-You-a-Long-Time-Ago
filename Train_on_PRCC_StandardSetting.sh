python3 tools/net_train.py --config_file='configs/STE_CNN_PRCC.yml' MODEL.PRETRAIN_PATH "../datasets_weights/pretrained_models" DATASETS.ROOT_DIR '../Long_term_datasets/prcc/prcc_nonID' Save_nonIDs_DIR "../Long_term_datasets/prcc/prcc_nonID/STE_features"


python3 tools/net_train.py --config_file='configs/LTE_CNN_PRCC_StandardSetting.yml' MODEL.PRETRAIN_PATH "../datasets_weights/pretrained_models" DATASETS.feat2_DIR '../Long_term_datasets/prcc/prcc_nonID/STE_features' DATASETS.ROOT_DIR "../Long_term_datasets/prcc/prcc_orig" DATASETS.CAMERA "B" OUTPUT_DIR '/OUTPUT/LTE_CNN_prcc_orig_shu2021'
