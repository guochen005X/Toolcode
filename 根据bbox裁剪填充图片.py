"""
输入bbox的参数（LeftTop_x,LeftTop_y,w,h)
根据bbox的参数裁剪图片，然后填充到新的图片，并保证bbox的中心点在新图片的中心
"""
import cv2
import numpy as np

def crop_image(bbox_param,origin_image):
    image_w = origin_image.shape[1]
    image_h = origin_image.shape[0]
    crop_left = int( bbox_param[0] + bbox_param[2]/2 - image_w / 2.0 )
    crop_left_padding = 0
    if crop_left < 0:
        crop_left_padding = -crop_left
        crop_left = 0
    else:
        crop_left_padding = 0

    crop_right = int( bbox_param[0] + bbox_param[2]/2 + image_w / 2.0 )
    if crop_right > image_w:
        crop_right = image_w

    crop_top_padding = 0
    crop_top  = int( bbox_param[1] + bbox_param[3]/2 - image_h / 2.0 )
    if crop_top < 0:
        crop_top_padding = -crop_top
        crop_top = 0
    else:
        crop_top_padding = 0

    crop_bottom = int( bbox_param[1] + bbox_param[3]/2 + image_h / 2.0 )
    if crop_bottom > image_h:
        crop_bottom = image_h

    new_imge = np.zeros([origin_image.shape[0],origin_image.shape[1], origin_image.shape[2]], dtype=np.uint8)
    im = origin_image[crop_top:crop_bottom, crop_left:crop_right, :]
    cv2.namedWindow('crop_image')
    cv2.imshow('crop_image', im)
    cv2.waitKey(0)

    new_imge[crop_top_padding:crop_top_padding + im.shape[0], crop_left_padding:crop_left_padding + im.shape[1], :] = im
    cv2.line(new_imge,(148,0),(148,288),(255,0,0))
    cv2.line(new_imge, (0, 149), (295, 149), (255, 0, 0))
    cv2.namedWindow('NEW_image')
    cv2.imshow('NEW_image', new_imge)
    cv2.waitKey(0)



if __name__ == '__main__':
    img_dir = 'G:\\DATESETS\\VOCdevkit\\test1\\2.jpg'
    img = cv2.imread(img_dir)
    cv2.namedWindow('ORIGIN_image')
    print('高 = {0}   宽 = {1}'.format(img.shape[0],img.shape[1]))

    cv2.rectangle(img, (50, 50), (100, 100), (0, 0, 255))

    cv2.imshow('ORIGIN_image',img)
    cv2.waitKey(0)
    bbox = [50,50,50,50]
    crop_image(bbox,img)




