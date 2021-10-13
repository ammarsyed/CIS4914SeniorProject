import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img = cv2.imread("C:/Users/Glados/Desktop/Peanut_MR/test/masks_pixel_gt//Peanut_640x480_DPI120/GT_Peanut_T003_L015_2017.06.15_080652_EED_DPI120.png", 1)
    img = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thinned = cv2.ximgproc.thinning(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    sobel = cv2.Sobel(src=thinned, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    cv2.imshow("sobel", sobel)

    edges = cv2.Canny(image=thinned, threshold1=100, threshold2=200)
    cv2.imshow('Canny Edge Detection', edges)

    cv2.imshow("skeleton", thinned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
