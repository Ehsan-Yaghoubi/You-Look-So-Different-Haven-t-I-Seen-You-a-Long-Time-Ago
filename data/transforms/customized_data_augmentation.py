"""
@author:  Ehsan Yaghoubi
@contact: Ehsan.Yaghoubi@gmail.com
"""

# from torchvision.transforms.functional import to_tensor
import random
from _01_preprocessing_step.generate_synthetic_images import online_replace_img1_back_with_others_back
from _01_preprocessing_step.generate_synthetic_images import offline_replace_img1_back_with_others_back

from PIL import Image
import os
import cv2


def read_image(img_path):
    """ Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process. """
    img = None
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class PartSubstitution(object):

    def __init__(self, probability, MaskDir, ImgDir, target_background_dir, constraint_funcs, other_attrs, online_image_processing_for_each_image, online_image_processing_for_all_images_once, TargetImagesArray):

        self.probability = probability
        self.MaskDir = MaskDir
        self.ImgDir = ImgDir
        self.constraint_funcs = constraint_funcs
        self.other_attrs = other_attrs
        self.online_image_processing_for_each_image = online_image_processing_for_each_image
        self.online_image_processing_for_all_images_once = online_image_processing_for_all_images_once
        self.target_background_dir = target_background_dir
        self.TargetImagesArray = TargetImagesArray
    def __call__(self, current_image_path):

        img = read_image(current_image_path)
        if random.uniform(0, 1) >= self.probability:
            return img

        img_name = current_image_path.split("/")[-1]

        if self.online_image_processing_for_each_image:
            img = online_replace_img1_back_with_others_back (name_img1 = img_name, MaskDir = self.MaskDir, ImgDir = self.ImgDir, enable_constraints_ht=  self.constraint_funcs)
        elif self.online_image_processing_for_all_images_once:
            img = offline_replace_img1_back_with_others_back (name_img1 = img_name, MaskDir1 = self.MaskDir, ImgDir1 = self.ImgDir, target_background_array = self.TargetImagesArray, target_background_dir= None)   ## Disadvantage is that it is not possible to consider IoU constrain to see if the background is suitable enough or not.
        else:
            img = offline_replace_img1_back_with_others_back (name_img1 = img_name, MaskDir1 = self.MaskDir, ImgDir1 = self.ImgDir, target_background_array = None, target_background_dir= self.target_background_dir)   ## Disadvantage is that it is not possible to consider IoU constrain to see if the background is suitable enough or not.
        #assert img is not None # The processed image is None! check the paths to the images.
        return img
