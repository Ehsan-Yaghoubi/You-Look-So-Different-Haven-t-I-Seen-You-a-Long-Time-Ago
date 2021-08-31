python3 tools/net_train.py --config_file='configs/STE_CNN_NKUP.yml' MODEL.PRETRAIN_PATH '../datasets_weights/pretrained_models' DATASETS.ROOT_DIR '../Long_term_datasets/nkup/nkup_nonID' Save_nonIDs_DIR '../Long_term_datasets/nkup/nkup_nonID/STE_features'


python3 tools/net_train.py --config_file='configs/LTE_CNN.yml' MODEL.PRETRAIN_PATH '../datasets_weights/pretrained_models'  DATASETS.feat2_DIR '../Long_term_datasets/nkup/nkup_nonID/STE_features' DATASETS.ROOT_DIR '../Long_term_datasets/nkup/nkup_orig' OUTPUT_DIR '/OUTPUT/LTE_CNN_nkup'
