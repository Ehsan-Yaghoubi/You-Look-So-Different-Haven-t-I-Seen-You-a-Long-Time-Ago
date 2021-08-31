# You Look So Different! Haven’t I Seen You a Long Time Ago?

## requirments:
- python == 3.5
- tensorflow == 1.13.2 (To use Hifill image painting)
- torch == 1.4.0
- trochvision == 0.5.0
- pytorch-ignite == 0.1.2

## Preprocessing:
1. extract [body-keypoints](https://colab.research.google.com/drive/1-invDDFpyVFlVuJSAV6AWyZgh4rNc3vF?usp=sharing) and body masks (using mask_extraction.py)
2. prepare the noneID dataset by running `Preprocessing.py`

You can skip the above steps, as we have provided the processed data at the links below (Please ask for the original data form the provider of each dataset):

[LTCC data: 1.2 GB](https://drive.google.com/file/d/1g2CKswZFDxovJinkEDvfoSaqpRuw3qAB/view?usp=sharing)

[PRCC data: 0.58 GB](https://drive.google.com/file/d/1lk51Yz8WJ_X79Jo5S-cTDSp6nIfHO1tb/view?usp=sharing)

[NKUP data: 0.43 GB](https://drive.google.com/file/d/1Jx11hRuFJAm60wNVQrWReXhVBOoC9fRb/view?usp=sharing)

## Train and Test the LSD Model

You can use (1) `.sh` scripts to train and test the method or (2) you can use an IDE and run `train.py` and `test.py` files.
In the first case, remember to set the directories to the weights and data in all the following `.sh` files correctly, and in the latter case, please set the variables and directories in `.yml` files at `configs` path.

### (1) Use .sh files to train and test the model

For the LTCC dataset, first run `sh Train_on_LTCC.sh` and then run `sh Test_on_LTCC.sh`. 

For the PRCC dataset in standard setting, first run `sh Train_on_PRCC_StandardSetting.sh` and then run `sh Test_on_PRCC.sh`, and for the cloth changing setting, first run `sh Train_on_PRCC_ClothChangingSetting.sh` and then run `sh Test_on_PRCC.sh`.

NKUP dataset 
First run `sh Train_on_NKUP.sh` and then run `sh Test_on_NKUP.sh`.

### (2) Use IDE to train and test the model
To train, in `tools/train.py`, line 90, set the `--config_file` to one of `.yml` files you want. E.g., `LTE_CNN_PRCC_ClothChangingSetting.yml`, and remember to modify the paths in `./configs/LTE_CNN_PRCC_ClothChangingSetting.yml`.
To test the model, in `tools/test.py`, line 17, set the `--config_file` to the `.yml` file you want. E.g., `LTE_CNN_PRCC_ClothChangingSetting.yml`, and remember to modify the model weight path in `./configs/LTE_CNN_PRCC_ClothChangingSetting.yml`.

# Reference:
In case our project helps your research, we appreciate it if you cite it in you works.
