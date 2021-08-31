from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import random
import torch

"""
This _01_preprocessing_step allows you to extrac full body masks of people using MaskRCNN
 Before run please set 'main_path' and 'dir_to_save' variables.
 """

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def load_image(image_name, transform_img=False):
    image = cv2.imread(image_name)
    if image is None:
        return False
    if transform_img:
        image = apply_brightness_contrast(input_img=image, brightness=random.randint(0,250)-125, contrast=random.randint(0,250)-125)
    image = cv2.copyMakeBorder(image, 200, 200, 250, 250, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    image = Image.fromarray(image)
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to('cuda')
    return image

def get_prediction(img_path, threshold):

    # image = cv2.imread(img_path)
    # image = cv2.copyMakeBorder(image, 200, 200, 250, 250, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # img = Image.fromarray(image)
    # # img = Image.open(img_path)
    # transform = T.Compose([T.ToTensor()])
    # img = transform(img)


    image = load_image(img_path)
    if image is False:
        return False, False, False, False

    with torch.no_grad():
        pred = model(image)
        # pred[0].to('cpu')
        if len(pred[0]['masks']) == 0:
            for i in range (20):
                image = load_image(img_path, transform_img=True)
                pred = model(image)
                if len(pred[0]['masks']) != 0: break
            if len(pred[0]['masks']) == 0:
                print("masks is NONE. masks is NONE. masks is NONE ...")
                return None, None, None, None
        if len(pred[0]['scores']) == 0:
            print("SCORE is NONE. SCORE is NONE. SCORE is NONE ...")
            return None, None, None, None

        pred_score = list(pred[0]['scores'].to('cpu').detach().numpy())
        masks = (pred[0]['masks'].to('cpu') > 0.5).squeeze(axis=1).detach().cpu().numpy()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].to('cpu').numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].to('cpu').detach().numpy())]
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
        if len(pred_t) == 0:
            for i in range (20):
                image = load_image(img_path, transform_img=True)
                pred = model(image)
                pred_score = list(pred[0]['scores'].to('cpu').detach().numpy())
                pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
                if len(pred_t) != 0: break

            print("ATENTION ............... pred_t is none")
            return None, None, None, None
        pred_t = pred_t[-1]
        masks = masks[:pred_t + 1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return masks, pred_boxes, pred_class, pred_score


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def choose_one_mask_of_person(masks, pred_score, pred_cls):
    for index, clas in enumerate(pred_cls):
        if clas == "person":
            pred_cls[index] = 1
        else:
            pred_cls[index] = 0

    mask_areas = []
    for index, mask in enumerate(masks):
        int_mas = mask * 1
        mask_areas.append(cv2.countNonZero(int_mas))
    mask_areas = np.array(mask_areas)
    pred_score_arr = np.array(pred_score[0:len(masks)])
    mask_scores = np.array(pred_cls) * pred_score_arr

    estimate = (mask_areas * mask_scores) / (mask_areas + mask_scores)
    max_score = np.nanmax(estimate)
    max_index = list(estimate).index(max_score)
    Mask = masks[max_index]
    return Mask


def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    # this function shows all masks of all categories
    masks, boxes, pred_cls, pred_score = get_prediction(img_path, threshold)
    if masks is False:
        return False
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    rgb_mask = random_colour_masks(masks)
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def instance_person_segmentation(img_path, threshold=0.5, display=False, dir_to_save=None):
    # this function selects, shows, and saves only one mask of the person class
    Masks, boxes, pred_cls, pred_score = get_prediction(img_path, threshold)

    if Masks is False:  # if no mask is detected
        return False

    if Masks is None:  # if no mask is detected
        return None

    if Masks.shape[0] == 1:  # if one mask is detected
        mask = np.squeeze(Masks, axis=0)
        h, w = mask.shape

    if Masks.shape[0] > 1:  # if several masks are detected
        mask = choose_one_mask_of_person(Masks, pred_score, pred_cls)
        h, w = mask.shape

    mask = mask[200:h - 200, 250:w - 250]
    if display:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_mask = random_colour_masks(mask)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    if dir_to_save is not None:
        file_name = os.path.basename(img_path)
        file_name = file_name
        try:
            cv2.imwrite(os.path.join(dir_to_save, file_name), mask * 255)
        except cv2.error:
            print(mask)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.cuda()

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    main_path = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/NKUP/NKUP_orig"
    dir_to_save = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/NKUP/NKUP_masks"

    ## make only one of the follwoing variables True, and the rest False.
    Folders_of_folders_of_folders_of_images = False
    Folders_of_folders_of_images = True
    Folders_of_images = False

    if Folders_of_images:
        ### Run on a folder of images
        list_of_images = os.listdir(main_path)
        for index3, image in enumerate(list_of_images):
            if index3 % 50 == 0:
                print(">>> mask extraction. Number of images: \t{}/{}".format(index3, len(list_of_images)))
            os.makedirs(dir_to_save, exist_ok=True)

            if os.path.isfile(os.path.join(dir_to_save, image)):
                print("mask exists")
            else:
                instance_person_segmentation(img_path=os.path.join(main_path, image),
                                             threshold=0.5,
                                             display=False,
                                             dir_to_save=dir_to_save)


    if Folders_of_folders_of_images:
        flag = 0
        notread = 0
        ##### Run on folders of folders of images
        list_folders = os.listdir(main_path)
        for i1, f1 in enumerate(list_folders):
            list_of_images = os.listdir(os.path.join(main_path, f1))
            for i2, image in enumerate(list_of_images):
                os.makedirs(os.path.join(dir_to_save, f1), exist_ok=True)
                if os.path.isfile(os.path.join(dir_to_save, f1, image)):
                    print("mask exists")
                else:
                    out = instance_person_segmentation(img_path=os.path.join(main_path, f1, image),
                                                 threshold=0.5,
                                                 display=False,
                                                 dir_to_save=os.path.join(dir_to_save, f1))
                    if out is False:
                        notread += 1
                        print("Image is not readable: ", notread)
                        continue
                    if out is None:
                        flag += 1
                        print ("The image is read but no mask is detected: ", flag)

            print(">>> mask extraction. Number of folders done: \t{}/{}".format(i1, len(list_folders)))

    if Folders_of_folders_of_folders_of_images:
        ##### Run on folders of folders of folders if images
        paren_folders = os.listdir(main_path)
        for i0, f0 in enumerate(paren_folders):
            list_folders = os.listdir(os.path.join(main_path, f0))
            for i1, f1 in enumerate(list_folders):
                list_of_images = os.listdir(os.path.join(main_path, f0, f1))
                for i2, image in enumerate(list_of_images):
                    os.makedirs(os.path.join(dir_to_save, f0, f1), exist_ok=True)
                    if os.path.isfile(os.path.join(dir_to_save, f0, f1, image)):
                        print("mask exists")
                    else:
                        instance_person_segmentation(img_path=os.path.join(main_path, f0, f1, image),
                                                     threshold=0.5,
                                                     display=False,
                                                     dir_to_save=os.path.join(dir_to_save, f0, f1))
                if i1 % 5 == 0:
                    print(">>> mask extraction. Number of folders done: \t{}/{}".format(i1, len(list_folders)))