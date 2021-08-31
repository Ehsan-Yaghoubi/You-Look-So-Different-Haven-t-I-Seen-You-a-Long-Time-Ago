python3 tools/net_train.py --config_file='configs/STE_CNN_LTCC.yml' MODEL.PRETRAIN_PATH '../datasets_weights/pretrained_models' DATASETS.ROOT_DIR '../datasets_weights/Long_term_datasets/LTCC_ReID/LTCC_inpainted' Save_nonIDs_DIR '../Long_term_datasets/LTCC_ReID/LTCC_nonID/STE_features' OUTPUT_DIR '/OUTPUT/STE_CNN_ltcc'


python3 tools/net_train.py --config_file='configs/LTE_CNN_LTCC.yml' MODEL.PRETRAIN_PATH '../datasets_weights/pretrained_models' DATASETS.feat2_DIR '../Long_term_datasets/LTCC_ReID/ltcc_nonID/STE_features' DATASETS.ROOT_DIR '../Long_term_datasets/LTCC_ReID/LTCC_orig'  OUTPUT_DIR '/OUTPUT/LTE_CNN_ltcc'
