
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import random
from sample_imageinpainting_HiFill.GPU_CPU import HiFill_inpainting
import wand
from wand.image import Image

"""
# This _01_preprocessing_step receives the mask and key points of person images and removes
# some biometric information (i.e., shape, head area, geometric information)

To extract the body masks use the '0_mask_extraction.py' _01_preprocessing_step.
To extract body kepoints use https://colab.research.google.com/drive/1-invDDFpyVFlVuJSAV6AWyZgh4rNc3vF?usp=sharing 
"""

def load_keypoints(json_dir, img_name, img_arry, mask, display=False):
    abs_name = img_name.split(".")[0]
    json_name = abs_name + ".json"
    try:
        with open(os.path.join(json_dir, json_name)) as json_file:
            data = json.load(json_file)
            _keypoints = data['bodies'][0]['joints']
    except FileNotFoundError:
        return None
    if display:
        display_keypoints(_keypoints, img_arry)
        cv2.imshow("img", img_arry)
        cv2.imshow("msk", mask)
        cv2.waitKey(0)
    return _keypoints


def load_mask(_masks_dir_, _img_name, display=False):
    _body_mask_array = cv2.imread(os.path.join(_masks_dir_, _img_name), cv2.IMREAD_GRAYSCALE)
    if display:
        cv2.imshow("_body_mask_", _body_mask_array)
        cv2.waitKey(0)
    return _body_mask_array


def load_img(_image_dir_, _img_name_, display=False):
    _img_array = cv2.imread(os.path.join(_image_dir_, _img_name_))
    if display:
        cv2.imshow("orig_img", _img_array)
        cv2.waitKey(0)
    return _img_array


def display_keypoints(keypoints, img_arry):
    """
    keypoint 0 : nose :             keypoints[0:3]
    keypoint 1 : neck :             keypoints[3:6]
    keypoint 2 : right shoulder :   keypoints[6:9]
    keypoint 3 : right elbow :      keypoints[9:12]
    keypoint 4 : right hand :       keypoints[12:15]
    keypoint 5 : left shoulder :    keypoints[15:18]
    keypoint 6 : left elbow :       keypoints[18:21]
    keypoint 7 : left hand :        keypoints[21:24]
    keypoint 8 : right hip :        keypoints[24:27]
    keypoint 9 : right knee :       keypoints[27:30]
    keypoint 10 : right foot :      keypoints[30:33]
    keypoint 11 : left hip :        keypoints[33:36]
    keypoint 12 : left knee :       keypoints[36:39]
    keypoint 13 : left foot :       keypoints[39:42]
    keypoint 14 : right eye :       keypoints[42:45]
    keypoint 15 : left eye :        keypoints[45:49]
    keypoint 16 : right ear :       keypoints[49:51]
    keypoint 17 : left ear :        keypoints[51:53]
    """
    for i in range(0, 54, 3):
        # round(255 * (keypoints[i + 2]))
        if (keypoints[i + 2]) >= 0.8:
            color = (0, 255, 0)
        elif 0.6 <= (keypoints[i + 2]) < 0.8:
            color = (0, 125, 0)
        elif 0.3 < (keypoints[i + 2]) < 0.6:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(img=img_arry,
                   center=(round(keypoints[i]), round(keypoints[i + 1])),
                   radius=4,
                   color=color,
                   thickness=-1)
        if i == 53:
            break


def get_head_mask(orig_mask, keypoints, display=False):
    # we can take the roi above the shoulders
    shoulder_r = keypoints[6:9]
    shoulder_l = keypoints[15:18]
    roi = [0,
           0,
           orig_mask.shape[0],
           round(max(shoulder_l[1], shoulder_r[1]))]  # x1, y1, x2, y2
    roi_img = np.copy(orig_mask[roi[1]:roi[3], roi[0]: roi[2]])  # use numpy slicing to get the image of the head
    head_mask = np.zeros(orig_mask.shape, np.uint8)  # make all the mask black
    head_mask[roi[1]:roi[3], roi[0]: roi[2]] = roi_img  # copy back only the roi

    kernel = np.ones((5, 5), np.uint8)
    head_mask = cv2.dilate(head_mask, kernel)

    if display:
        cv2.imshow("new_mask", head_mask)
        cv2.waitKey(0)

    return head_mask


def get_mask_boarder(orig_mask, display=False):
    dilated_mask = mask_dilation_erosion(orig_mask, 'dilate', proportion=0.03, _iter=1)
    eroded_mask = mask_dilation_erosion(orig_mask, 'erode', proportion=0.033, _iter=2)

    # eroded_mask = ~eroded_mask
    eroded_mask = cv2.bitwise_not(eroded_mask)
    boarder_area = cv2.bitwise_and(dilated_mask, eroded_mask)
    if display:
        cv2.imshow("boarder_area", boarder_area)
        cv2.waitKey(0)

    return boarder_area, dilated_mask, eroded_mask


def get_final_mask(head_mask, mask_boarder, display=False):
    # assert head_mask.shape==mask_boarder.shape
    # assert head_mask.dtype==mask_boarder.dtype
    final_mask = cv2.bitwise_or(head_mask, mask_boarder)
    if display:
        cv2.imshow("final_mask", final_mask)
        cv2.waitKey(0)
    return final_mask


def do_inpainting(img, mask, display=False):
    radius = int(max(np.array(mask.shape)) * 0.1)
    inpainted_img = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
    if display:
        cv2.imshow("inpainted_img", inpainted_img)
        cv2.waitKey(0)
    return inpainted_img


def mask_dilation_erosion(mask, flag, proportion=0.03, _iter=1):
    kernel_size = (np.array(mask.shape) * proportion).astype(int)
    kernel = np.ones(kernel_size, np.uint8)
    if flag == 'dilate':
        _mask = cv2.dilate(mask, kernel, iterations=_iter)
    elif flag == 'erode':
        _mask = cv2.erode(mask, kernel, iterations=_iter)
    else:
        print("Specify the flag correctly")
        raise ValueError

    return _mask


def perturbed_mesh(row, column, desplay=False):
    # the idea has been taken from paper https://www.juew.org/publication/DocUNet.pdf
    mr = row
    mc = column

    xx = np.arange(mr - 1, -1, -1)
    yy = np.arange(0, mc, 1)

    # xx1 = np.random.randint(0,mr,(1,mr))[0]
    # yy1 =np.random.randint(0,mc,(1,mc))[0]

    [y, X] = np.meshgrid(xx, yy)
    X_flatten = X.flatten('F')
    Y_flatten = y.flatten('F')
    XY_mat = [X_flatten, Y_flatten]
    XY_mat_arr = np.asarray(XY_mat)
    ms = np.transpose(XY_mat_arr, (1, 0))

    perturbed_mesh_ = ms
    nv = np.random.randint(20) - 1
    for k in range(nv):
        # Choosing one vertex randomly
        vidx = np.random.randint(np.shape(ms)[0])
        vtex = ms[vidx, :]
        # Vector between all vertices and the selected one
        xv = perturbed_mesh_ - vtex
        # Random movement
        mv = (np.random.rand(1, 2) - 0.5) * 20
        hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] + 1))
        hxv[:, :-1] = xv
        hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0], 1))
        d = np.cross(hxv, hmv)
        d = np.absolute(d[:, 2])
        d = d / (np.linalg.norm(mv, ord=2))
        wt = d

        curve_type = np.random.rand(1)
        if curve_type > 0.3:
            alpha = np.random.rand(1) * 50 + 50
            wt = alpha / (wt + alpha)
        else:
            alpha = np.random.rand(1) + 1
            wt = 1 - (wt / 100) ** alpha
        msmv = mv * np.expand_dims(wt, axis=1)
        perturbed_mesh_ = perturbed_mesh_ + msmv
    if desplay:
        plt.scatter(perturbed_mesh_[:, 0], perturbed_mesh_[:, 1], c=np.arange(0, mr * mc))
        plt.show()
    return perturbed_mesh_[:, 0], perturbed_mesh_[:, 1]


def make_same_sizes(img1, img2, img2_mask):
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]
    if h1 > h2:  # increase the height of img2
        img2_p = cv2.copyMakeBorder(img2, int((h1 - h2) / 2), int((h1 - h2) / 2), 0, 0, borderType=cv2.BORDER_REFLECT)
        img2_m_p = cv2.copyMakeBorder(img2_mask, int((h1 - h2) / 2), int((h1 - h2) / 2), 0, 0,
                                      borderType=cv2.BORDER_REFLECT)
        img1_p = np.copy(img1)
    elif h2 > h1:  # increase the height of img1
        img2_p = np.copy(img2)
        img2_m_p = np.copy(img2_mask)
        img1_p = cv2.copyMakeBorder(img1, int((h2 - h1) / 2), int((h2 - h1) / 2), 0, 0, borderType=cv2.BORDER_REFLECT)
    else:
        img1_p = np.copy(img1)
        img2_p = np.copy(img2)
        img2_m_p = np.copy(img2_mask)

    if w1 > w2:  # increase the width of img2
        img2_p = cv2.copyMakeBorder(img2_p, 0, 0, int((w1 - w2) / 2), int((w1 - w2) / 2), borderType=cv2.BORDER_REFLECT)
        img2_m_p = cv2.copyMakeBorder(img2_m_p, 0, 0, int((w1 - w2) / 2), int((w1 - w2) / 2),
                                      borderType=cv2.BORDER_REFLECT)
    elif w2 > w1:  # increase the width of img1
        img1_p = cv2.copyMakeBorder(img1_p, 0, 0, int((w2 - w1) / 2), int((w2 - w1) / 2), borderType=cv2.BORDER_REFLECT)
    else:
        pass

    if img1_p.shape[0:2] == img2_p.shape[0:2] == img2_m_p.shape[0:2]:
        return img1_p, img2_p, img2_m_p
    else:  # make them the same size
        # h,w = img1_p.shape[0:2]
        img1_p = cv2.resize(img1_p, (w1, h1))
        img2_p = cv2.resize(img2_p, (w1, h1))
        img2_m_p = cv2.resize(img2_m_p, (w1, h1))
        return img1_p, img2_p, img2_m_p


def do_deformed_mesh(background, img, mask, display=False):
    nw, nh = img.shape[0:2]
    dw = nw // 2
    dh = nh // 2

    gray_mask = np.zeros(mask.shape[0:2], dtype=np.uint8)
    gray_mask[:, :] = mask[:, :, 0]
    body_area = cv2.bitwise_and(img, img, mask=gray_mask)

    # when performing the deformation, sometimes the some parts of the RoI are placed out of the image boarders.
    # Therefore, to avoid it, we first enlarge the image and its mask.
    extended_body_area = cv2.copyMakeBorder(body_area, dh, dh, dw, dw, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    extended_body_area_mask = cv2.copyMakeBorder(gray_mask, dh, dh, dw, dw, borderType=cv2.BORDER_CONSTANT,
                                                 value=(0, 0, 0))

    nw, nh = extended_body_area.shape[0:2]

    xs, ys = perturbed_mesh(nh, nw, False)  # the result is like np.meshgrid
    xs = xs.reshape(nh, nw).astype(np.float32)
    ys = ys.reshape(nh, nw).astype(np.float32)

    deformed_extended_img = cv2.remap(extended_body_area, ys, xs, cv2.INTER_CUBIC)
    deformed_extended_img = cv2.rotate(deformed_extended_img, cv2.ROTATE_90_CLOCKWISE)

    deformed_extended_msk = cv2.remap(extended_body_area_mask, ys, xs, cv2.INTER_CUBIC)
    deformed_extended_msk = cv2.rotate(deformed_extended_msk, cv2.ROTATE_90_CLOCKWISE)
    # minimum rectangular contour (We capture the deformed RoI and then we will make it the same size of the original
    # image. Then, we past it on the original background)
    deformed_image = cv2.cvtColor(deformed_extended_img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(deformed_image)

    # rectangular contour of image ==> should be resized to original image size
    deformed_image = deformed_extended_img[y:y + h, x:x + w, :]
    # rectangular contour of mask ==> should be resized to original image size
    deformed_mask = deformed_extended_msk[y:y + h, x:x + w]

    # background variable has the same size of the original image
    bakgrand, deformed_image, deformed_mask = make_same_sizes(background, deformed_image, deformed_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.erode(deformed_mask, kernel)
    mask_3 = np.expand_dims(mask2, axis=2)
    mask4 = mask_3 * np.ones((1, 1, 3))

    background1 = np.copy(bakgrand)
    background2 = np.copy(bakgrand)

    seed = random.randint(0, 1000000000)
    cropped_dst_scratched = do_scratch(deformed_image, 0.85, 1.8, seed, display)
    mask4_scratched = do_scratch(mask4, 0.85, 1.8, seed, display)
    mask4_scratched_bool = np.copy(mask4_scratched)
    mask4_scratched_bool = mask4_scratched_bool > 127
    np.copyto(background1, cropped_dst_scratched, casting='unsafe', where=mask4_scratched_bool)

    mask4 = mask4 > 127
    np.copyto(background2, deformed_image, casting='unsafe', where=mask4)

    if display:
        cv2.imshow("scratched", background1)
        cv2.imshow("No_scratched", background2)
        cv2.imshow("mask_No_scratched", mask2)
        cv2.imshow("mask_scratched", mask4_scratched)
        cv2.waitKey(0)

    return background1, mask4_scratched, background2, mask2


def do_scratch(img, min_prop, max_prop, seed, display=False):
    # this function scratches the input image and return the output with the same size
    if min_prop >= 1:
        raise ValueError("select a value LESS than 1")
    if max_prop <= 1:
        raise ValueError("select a value MORE than 1")

    h1, w1 = img.shape[0:2]
    random.seed(seed)
    w2 = random.randint(int(min_prop * w1), int(max_prop * w1))
    h2 = random.randint(int(min_prop * h1), int(max_prop * h1))
    resized_img = cv2.resize(img, (w2, h2))

    p1 = h1 / w1
    p2 = h2 / w2

    if p1 > p2:
        add_to_length = (w2 * p1) - h2
        resized_img = cv2.copyMakeBorder(resized_img, int(add_to_length / 2), int(add_to_length / 2), 0, 0,
                                         cv2.BORDER_CONSTANT, (0, 0, 0))
        resized_img = cv2.resize(resized_img, (w1, h1))
        assert img.shape == resized_img.shape
        if display:
            cv2.imshow("resized_img", resized_img)
            cv2.waitKey(0)
        return resized_img
    elif p2 > p1:
        add_to_width = (h2 / p1) - w2
        resized_img = cv2.copyMakeBorder(resized_img, 0, 0, int(add_to_width / 2), int(add_to_width / 2),
                                         cv2.BORDER_CONSTANT, (0, 0, 0))
        resized_img = cv2.resize(resized_img, (w1, h1))
        assert img.shape == resized_img.shape

        if display:
            cv2.imshow("resized_img", resized_img)
            cv2.waitKey(0)
        return resized_img
    else:
        resized_img = cv2.resize(resized_img, (w1, h1))
        assert img.shape == resized_img.shape

        if display:
            cv2.imshow("resized_img", resized_img)
            cv2.waitKey(0)
        return resized_img


def write_img(_dir, img_array, img_name):
    os.makedirs(_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(_dir, img_name)):
        cv2.imwrite(os.path.join(_dir, img_name), img_array)
    else:
        print("file exists and did not over-writen")


def do_distorstion(img_array, mask_array, seed, img_name, display=False, save=False):
    # img_name = os.path.basename(dir_to_img)
    img = Image.from_array(img_array)
    msk = Image.from_array(mask_array)

    # with Image(filename=dir_to_img) as img:
    img.virtual_pixel = 'mirror'
    random.seed(seed)
    rand = random.uniform(-0.6, 0.8)

    img.implode(rand)
    if save:
        img.save(filename='{}_imploded_img.jpg'.format(img_name))
    # convert to opencv/numpy array format
    img = np.array(img)
    # image_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Perform the same operation on mask
    msk.virtual_pixel = 'mirror'
    random.seed(seed)
    rand = random.uniform(-0.6, 0.8)
    msk.implode(rand)
    if save:
        msk.save(filename='{}_imploded_mask.jpg'.format(img_name))
    # convert to opencv/numpy array format
    msk = np.array(msk)
    # mask_ = cv2.cvtColor(msk, cv2.COLOR_RGB2BGR)

    # display result with opencv
    if display:
        cv2.imshow("imploded_Image", img)
        cv2.imshow("imploded_Mask", msk)
        cv2.waitKey(0)

    return img, msk


def transform_one_image():
    source_dir = "../images_for_test"
    dir_to_save_noneID = "../images_for_test/processed"

    img1 = "img_001_1_c4_015857.png"
    maks1 = "mask_001_1_c4_015857.png"
    key1 = "001_1_c4_015857.json"
    img2 = "img_006_1_c4_000137.png"
    maks2 = "mask_006_1_c4_000137.png"
    key2 = "006_1_c4_000137.json"

    img_name = img2
    mask_name = maks2
    key = key2

    for i in range(1, 20, 1):
        dir_ = os.path.join(dir_to_save_noneID, str(i))
        os.makedirs(dir_, exist_ok=True)

        img_arry = load_img(source_dir, img_name)
        orig_mask = load_mask(source_dir, mask_name)
        keypoints = load_keypoints(source_dir, key, img_arry, orig_mask, False)

        background = HiFill_inpainting.HiFill(img_arry, orig_mask, False)
        write_img(dir_, background, "{}_background.jpg".format(img_name))

        head_mask = get_head_mask(orig_mask, keypoints, False)
        write_img(dir_, head_mask, "{}_head_mask.jpg".format(img_name))

        mask_boarder, dilated_mask, eroded_mask = get_mask_boarder(orig_mask, False)
        write_img(dir_, mask_boarder, "{}_boarder_area.jpg".format(img_name))
        write_img(dir_, dilated_mask, "{}_dilated_mask.jpg".format(img_name))
        write_img(dir_, eroded_mask, "{}_eroded_mask.jpg".format(img_name))

        final_mask = get_final_mask(head_mask, mask_boarder, False)
        write_img(dir_, final_mask, "{}_final_mask.jpg".format(img_name))

        HiFill_head_border = HiFill_inpainting.HiFill(img_arry, final_mask, False)
        write_img(dir_, HiFill_head_border, "{}_boarder_head_inpainted.jpg".format(img_name))

        seed = random.randint(1, 10000000000)
        imploded_img, imploded_mask = do_distorstion(HiFill_head_border, orig_mask, seed, img_name, display=False)
        write_img(dir_, imploded_img, "{}_imploded_img.jpg".format(img_name))
        write_img(dir_, imploded_mask, "{}_imploded_mask.jpg".format(img_name))

        img_deformed_scratched, mask_deformed_scratched, img_deformed_NOscratched, mask_deformed_NOscratched = do_deformed_mesh(
            background=background, img=imploded_img, mask=imploded_mask, display=False)  # can be done on mask and img
        write_img(dir_, img_deformed_scratched, "{}_img_deformed_scratched.jpg".format(img_name))
        write_img(dir_, mask_deformed_scratched, "{}_mask_deformed_scratched.jpg".format(img_name))
        write_img(dir_, img_deformed_NOscratched, "{}_img_deformed_NOscratched.jpg".format(img_name))
        write_img(dir_, mask_deformed_NOscratched, "{}_mask_deformed_NOscratched.jpg".format(img_name))


if __name__ == '__main__':
    # _img_name_ = "img_006_1_c4_000137.png"
    # img_array = cv2.imread(os.path.join('../images_for_test', _img_name_), cv2.COLOR_BGR2RGB)
    # img = Image.from_array(img_array)
    # with Image(img) as myimg:
    #     x = img_array.shape[1]
    #     y = img_array.shape[0]
    #     args = (
    #         x / 10, y / 10, x / 15, y / 15,  # Point 1: (10, 10) => (15,  15)
    #         5 * x / 10, 5 * y / 10, 5 * x / 15, 5 * y / 15,  # Point 2: (139, 0) => (100, 20)
    #         9 * x / 10, 9 * y / 10, 9 * x / 15, 9 * y / 15  # Point 3: (0,  92) => (50,  80)
    #     )
    #     myimg.distort('shepards', args)
    #     myimg.save(filename='../images_for_test/affineTransformedImage.png')

    # mask_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_masks"
    # image_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_orig"
    # json_files_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_keypoints"
    # dir_to_save_noneID = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_nonID_Last"

    # mask_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/prcc/prcc_masks/test"
    # image_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/prcc/prcc_orig/test"
    # json_files_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/prcc/prcc_keypoints/test"
    # dir_to_savenoneID = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/prcc/prcc_nonID_new/scratching/test"

    # mask_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/celeb_reid/celebreid_masks"
    # image_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/celeb_reid/celebreid_orig"
    # json_files_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/celeb_reid/celebreid_keypoints"
    # dir_to_save_noneID = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/celeb_reid/celebreid_nonID_new/scratching"


    # paper_fig_mask_dir = mask_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_masks/train"
    # paper_fig_image_dir = image_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/YLD_YouLookDifferent/images_for_test/images"
    # paper_fig_json_files_dir = json_files_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_keypoints/train"
    # paper_fig_dir_to_save_noneID = dir_to_save_noneID = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/A_PROJECTS/LOCAL/cvpr2021/YLD_YouLookDifferent/images_for_test/Transformed_images"

    mask_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/NKUP/NKUP_masks/query"
    image_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/NKUP/NKUP_orig/query"
    json_files_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/NKUP/NKUP_keypoints/query"
    dir_to_save_noneID = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/NKUP/NKUP_nonID/query"

    os.makedirs(dir_to_save_noneID, exist_ok=True)
    two_sub_folders = False
    one_sub_folder = False
    without_sub_folder = True
    One_time = False

    for repetition in range (11):
        if one_sub_folder:
            folder_names = os.listdir(mask_dir)
            for F_index, fld_name in enumerate(folder_names):
                #flag = 0
                images_names = os.listdir(os.path.join(mask_dir, fld_name))
                for index, img_name in enumerate(images_names):

                    if os.path.isfile(os.path.join(dir_to_save_noneID, fld_name, img_name)):
                        print("image {} is existing".format(img_name))
                        continue

                    img_arry = load_img(os.path.join(image_dir, fld_name), img_name)
                    orig_mask = load_mask(os.path.join(mask_dir, fld_name), img_name)
                    keypoints = load_keypoints(os.path.join(json_files_dir, fld_name), img_name, img_arry, orig_mask, False)
                    if keypoints is None or orig_mask is None or img_arry is None:
                        print("(img/mask/keypoint)data did not find: {}".format(img_name))
                        continue

                    background = HiFill_inpainting.HiFill(img_arry, orig_mask, False)
                    head_mask = get_head_mask(orig_mask, keypoints, False)
                    mask_boarder, dilated_mask, eroded_mask = get_mask_boarder(orig_mask, False)
                    final_mask = get_final_mask(head_mask, mask_boarder, False)
                    HiFill_head_border = HiFill_inpainting.HiFill(img_arry, final_mask, False)
                    seed = random.randint(1, 10000000000)
                    imploded_img, imploded_mask = do_distorstion(HiFill_head_border, orig_mask, seed, img_name, display=False)
                    img_deformed_scratched, mask_deformed_scratched, img_deformed_NOscratched, mask_deformed_NOscratched = do_deformed_mesh(
                        background=background, img=imploded_img, mask=imploded_mask,
                        display=False)  # can be done on mask and img

                    write_img(os.path.join(dir_to_save_noneID, fld_name),
                              img_deformed_scratched,
                              "_{}_{}".format(repetition, img_name))

                    if index % 100 == 0:
                        print("Save Generated Images to {}: {}/{}".format(
                            os.path.join(dir_to_save_noneID, fld_name), index, len(images_names)))

        if two_sub_folders:
            parent_fld = os.listdir(mask_dir)
            for p_index, p_fld_name in enumerate(parent_fld):
                folder_names = os.listdir(os.path.join(mask_dir, p_fld_name))
                flag = 0
                for F_index, fld_name in enumerate(folder_names):
                    flag2 = 0
                    images_names = os.listdir(os.path.join(mask_dir, p_fld_name, fld_name))
                    for index, img_name in enumerate(images_names):

                        if os.path.isfile(os.path.join(dir_to_save_noneID, p_fld_name, fld_name, img_name)):
                            print("image {} is existing".format(img_name))
                            continue

                        img_arry = load_img(os.path.join(image_dir, p_fld_name, fld_name), img_name)
                        orig_mask = load_mask(os.path.join(mask_dir, p_fld_name, fld_name), img_name)
                        keypoints = load_keypoints(os.path.join(json_files_dir, p_fld_name, fld_name), img_name, img_arry, orig_mask, False)
                        if keypoints is None or orig_mask is None or img_arry is None:
                            print("(img/mask/keypoint)data did not find: {}".format(img_name))
                            continue

                        background = HiFill_inpainting.HiFill(img_arry, orig_mask, False)
                        head_mask = get_head_mask(orig_mask, keypoints, False)
                        mask_boarder, dilated_mask, eroded_mask = get_mask_boarder(orig_mask, False)
                        final_mask = get_final_mask(head_mask, mask_boarder, False)
                        HiFill_head_border = HiFill_inpainting.HiFill(img_arry, final_mask, False)
                        seed = random.randint(1, 10000000000)
                        imploded_img, imploded_mask = do_distorstion(HiFill_head_border, orig_mask, seed, img_name, display=False)
                        img_deformed_scratched, mask_deformed_scratched, img_deformed_NOscratched, mask_deformed_NOscratched = do_deformed_mesh(
                            background=background, img=imploded_img, mask=imploded_mask,
                            display=False)  # can be done on mask and img

                        write_img(os.path.join(dir_to_save_noneID, p_fld_name, fld_name),
                                  img_deformed_scratched,
                                  "_{}_{}".format(repetition, img_name))

                        if index % 5 == 0:
                            print("Save Generated Images to {}: {}/{}".format(
                                os.path.join(dir_to_save_noneID, p_fld_name, fld_name), index, len(images_names)))

        if without_sub_folder:
            images_names = os.listdir(image_dir)
            for index, img_name in enumerate(images_names):

                if os.path.isfile(os.path.join(dir_to_save_noneID, img_name)):
                    print("image {} is existing".format(img_name))
                    continue

                img_arry = load_img(os.path.join(image_dir), img_name)
                orig_mask = load_mask(os.path.join(mask_dir), img_name)
                keypoints = load_keypoints(os.path.join(json_files_dir), img_name, img_arry, orig_mask, False)
                if keypoints is None or orig_mask is None or img_arry is None:
                    print("(img/mask/keypoint)data did not find: {}".format(img_name))
                    continue

                background = HiFill_inpainting.HiFill(img_arry, orig_mask, False)
                head_mask = get_head_mask(orig_mask, keypoints, False)
                mask_boarder, dilated_mask, eroded_mask = get_mask_boarder(orig_mask, False)
                final_mask = get_final_mask(head_mask, mask_boarder, False)
                HiFill_head_border = HiFill_inpainting.HiFill(img_arry, final_mask, False)
                seed = random.randint(1, 10000000000)
                imploded_img, imploded_mask = do_distorstion(HiFill_head_border, orig_mask, seed, img_name,
                                                             display=False)
                img_deformed_scratched, mask_deformed_scratched, img_deformed_NOscratched, mask_deformed_NOscratched = do_deformed_mesh(
                    background=background, img=imploded_img, mask=imploded_mask,
                    display=False)  # can be done on mask and img

                write_img(dir_to_save_noneID,
                          img_deformed_scratched,
                          "_{}_{}".format(repetition, img_name))

                if index % 100 == 0:
                    print("Save Generated Images to {}: {}/{}".format(
                        os.path.join(dir_to_save_noneID), index, len(images_names)))

        if One_time:
            print("if you want a loop for generating images, set the One_time variable to False.")
            break
