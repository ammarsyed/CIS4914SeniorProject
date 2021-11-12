import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    AmmarFilePath = "C:/Users/asyed/Documents/SeniorProjectDataset/Peanut_640x480_DPI120-20211013T024029Z-001/Peanut_640x480_DPI120/GT_Peanut_T032_L022_2017.07.27_113613_EED_DPI120.png"
    HannahFilePath = "/Users/hannahsoliman/Desktop/GT_Peanut_T032_L022_2017.07.27_113613_EED_DPI120.png"
    # HannahFilePath = "/Users/hannahsoliman/Desktop/CIS4914SeniorProject/GT_Peanut_T032_L029_2017.06.08_162402_EED_DPI120.png"
    NavidFilePath = "C:/Users/Glados/Desktop/Peanut_MR/test/masks_pixel_gt//Peanut_640x480_DPI120/GT_Peanut_T003_L015_2017.06.15_080652_EED_DPI120.png"

    img = cv2.imread(AmmarFilePath, 1)
    img = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thinned = cv2.ximgproc.thinning(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    #sobel = cv2.Sobel(src=thinned, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    img_sobelx = cv2.Sobel(thinned, cv2.CV_8U, 1, 0, ksize=5)
    img_sobely = cv2.Sobel(thinned, cv2.CV_8U, 0, 1, ksize=5)

    sobel = img_sobelx + img_sobely
    # cv2.imshow("sobel", sobel)

    # edges = cv2.Canny(image=thinned, threshold1=100, threshold2=200)
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.imshow("skeleton", thinned)

    # CONNECTED COMPONENT LABELING
    cv2.imwrite("SOBELONE.png", sobel)
    sobel_path = "SOBELONE.png"

    img = cv2.imread(sobel_path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(img)
    print("Number of roots: ", num_labels-1) #-1 to remove the background component
    # labels = cv2.connectedComponentsWithStats(img, stats=cv2.CV_32S)
    # (numLabels, labels, stats, centroids) = labels
    #
    # print(len(labels))

    # for i in range(0, numLabels):
    #     if i == 0:
    #         text = "examining component {}/{} (background)".format(
    #             i + 1, numLabels)
    #
    #     else:
    #         text = "examining component {}/{}".format(i + 1, numLabels)
    #
    #
    #     print("[INFO] {}".format(text))
    #
    #     x = stats[i, cv2.CC_STAT_LEFT]
    #     y = stats[i, cv2.CC_STAT_TOP]
    #     w = stats[i, cv2.CC_STAT_WIDTH]
    #     h = stats[i, cv2.CC_STAT_HEIGHT]
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     (cX, cY) = centroids[i]
    #
    #     labels = img.copy()
    #     cv2.rectangle(labels, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #     cv2.circle(labels, (int(cX), int(cY)), 4, (0, 0, 255), -1)
    #
    #     componentMask = (labels == i).astype("uint8") * 255
    #
    #     cv2.imshow("Output", labels)
    #     cv2.imshow("Connected Component", componentMask)
    #     cv2.waitKey(0)

    #coloring
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    labeled_img[label_hue == 0] = 0


    # Original
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Original Image")
    plt.show()

    # Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Component Labeling Image")
    plt.show()



    # cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
