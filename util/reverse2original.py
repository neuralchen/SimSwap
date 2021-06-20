import cv2
import numpy as np
# import  time
from util.add_watermark import  watermark_image

def reverse2wholeimage(swaped_imgs, mats, crop_size, oriimg, logoclass, save_path = '',):

    target_image_list = []
    img_mask_list = []
    for swaped_img, mat in zip(swaped_imgs, mats):
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size,crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2,3])
        div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
        mat_rev[0][0] = mat[1][1]/div1
        mat_rev[0][1] = -mat[0][1]/div1
        mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
        div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
        mat_rev[1][0] = mat[1][0]/div2
        mat_rev[1][1] = -mat[0][0]/div2
        mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
        img_white = cv2.warpAffine(img_white, mat_rev, orisize)


        img_white[img_white>20] =255

        img_mask = img_white

        kernel = np.ones((10,10),np.uint8)
        img_mask = cv2.erode(img_mask,kernel,iterations = 1)

        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
        target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255

        img_mask_list.append(img_mask)
        target_image_list.append(target_image)
    # target_image /= 255
    # target_image = 0
    img = np.array(oriimg, dtype=np.float)
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        img = img_mask * target_image + (1-img_mask) * img

    final_img = logoclass.apply_frames(img.astype(np.uint8))
    cv2.imwrite(save_path, final_img)

    # cv2.imwrite('E:\\lny\\SimSwap-main\\output\\img_div.jpg', img * 255)
    # cv2.imwrite('E:\\lny\\SimSwap-main\\output\\ori_img.jpg', oriimg)
    