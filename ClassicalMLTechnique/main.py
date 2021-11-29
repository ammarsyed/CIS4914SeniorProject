from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import os


class StegerLineAlgorithm:

    def computeDerivative(self, img, derivativeType):
        if derivativeType == 'dx':
                dxKernel = np.array([[1], [0], [-1]])
                dx = cv2.filter2D(img, -1, dxKernel)
                return dx
        elif derivativeType == 'dy':
                dyKernel = np.array([[1, 0, -1]])
                dy = cv2.filter2D(img, -1, dyKernel)
                return dy
        elif derivativeType == 'dxx':
                dxxKernel = np.array([[1], [-2], [1]])
                dxx = cv2.filter2D(img, -1, dxxKernel)
                return dxx
        elif derivativeType == 'dyy':
                dyyKernel = np.array([[1, -2, 1]])
                dyy = cv2.filter2D(img, -1, dyyKernel)
                return dyy
        elif derivativeType == 'dxy':
                dxyKernel = np.array([[1, -1], [-1, 1]])
                dxy = cv2.filter2D(img, -1, dxyKernel)
                return dxy

    def computeDerivatives(self, img, sigmaX, sigmaY):
        img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigmaX, sigmaY=sigmaY)

        dx = self.computeDerivative(img, 'dx')
        dy = self.computeDerivative(img, 'dy')
        dxx = self.computeDerivative(img, 'dxx')
        dyy = self.computeDerivative(img, 'dyy')
        dxy = self.computeDerivative(img, 'dxy')

        return dx, dy, dxx, dyy, dxy

    def computeMagnitude(self, dxx, dyy):
        dxx = dxx.astype(float)
        dyy = dyy.astype(float)
        mag = cv2.magnitude(dxx, dyy)
        phase = mag * 180. / np.pi
        return mag, phase

    def computeHessian(self, img, dx, dy, dxx, dyy, dxy):
        # create empty list
        point = []
        direction = []
        value = []
        # for the all image
        for x in range(0, img.shape[1]):  # column
            for y in range(0, img.shape[0]):  # line
                # if superior to certain threshold
                if dxy[y, x] > 0:
                    # compute local hessian
                    hessianMatrix = np.zeros((2, 2))
                    hessianMatrix[0, 0] = dxx[y, x]
                    hessianMatrix[0, 1] = dxy[y, x]
                    hessianMatrix[1, 0] = dxy[y, x]
                    hessianMatrix[1, 1] = dyy[y, x]
                    # compute eigen vector and eigen value
                    ret, eigenVal, eigenVect = cv2.eigen(hessianMatrix)
                    if np.abs(eigenVal[0, 0]) >= np.abs(eigenVal[1, 0]):
                        nx = eigenVect[0, 0]
                        ny = eigenVect[0, 1]
                    else:
                        nx = eigenVect[1, 0]
                        ny = eigenVect[1, 1]
                    # calculate denominator for the taylor polynomial expension
                    denom = dxx[y, x] * nx * nx + dyy[y, x] * ny * ny + 2 * dxy[y, x] * nx * ny
                    # verify non zero denom
                    if denom != 0:
                        T = -(dx[y, x] * nx + dy[y, x] * ny) / denom
                        # update point
                        if np.abs(T * nx) <= 0.5 and np.abs(T * ny) <= 0.5:
                            point.append((x, y))
                            direction.append((nx, ny))
                            value.append(np.abs(dxy[y, x] + dxy[y, x]))
        return point, direction, value

def initializeImage(img):
    # COMMENT FOLLOWING LINE IF USING UNET SEGMENTED ROOTS. UNCOMMENT IF USING GROUND TRUTH IMAGES
    # img = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thinned = cv2.ximgproc.thinning(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY),
                                    thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    img = thinned
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img


def getImageFromCommandLineArg():
    if len(sys.argv) == 1:
        print("requires image Source as an argument")
        sys.exit()

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("image filepath is invalid or doesn't exist")
        sys.exit()
    else:
        filePath = sys.argv[1]
        print(filePath)
        fileNameStartIndex = filePath.rfind("/")
        if(fileNameStartIndex == -1):
            fileNameStartIndex = filePath.rfind("\\")
        fileNameEndsWith = filePath.rfind(".")
        fileName = filePath[fileNameStartIndex+1:fileNameEndsWith]
        print(fileName)
        print(sys.argv[1])
        return fileName, img


def performConnectedComponentsAnalysis(image):
    test = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(test)
    print("Number of Connected Components excluding background: ", num_labels - 1)  # subtract 1 for the
    return num_labels, labels





def labelImage(labels):
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return labeled_img


def outputImages(fileName, labeled_img):
    # Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Component Labeling Image")
    if not os.path.exists("TraditionalLabeledImages"):
        os.makedirs("TraditionalLabeledImages")
    plt.savefig("TraditionalLabeledImages/TraditionalLabeled_" + fileName)
    plt.show()


def main():
    print("hello")
    fileName, img = getImageFromCommandLineArg()
    gray_img = initializeImage(img)

    # Curve Detection
    stegerLineAlgorithmObj = StegerLineAlgorithm()
    dx, dy, dxx, dyy, dxy = stegerLineAlgorithmObj.computeDerivatives(gray_img, 1.1, 1.1)
    normal, phase = stegerLineAlgorithmObj.computeMagnitude(dxx, dyy)

    pt, dir, val = stegerLineAlgorithmObj.computeHessian(img, dx, dy, dxx, dyy, dxy)
    idx = np.argsort(val)
    idx = idx[::-1][:10000]
    for i in range(0, len(idx)):
        img = cv2.circle(img, (pt[idx[i]][0], pt[idx[i]][1]), 1, (255, 0, 0), 1)
    outputImages(fileName, img)


    # Conversion
    cv2.imwrite('Test.png', phase)
    test = cv2.imread('Test.png')
    os.remove('Test.png')

    blur = cv2.GaussianBlur(test, (5, 5), 0)
    test = cv2.ximgproc.thinning(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    num_labels, labels = performConnectedComponentsAnalysis(test)

    labeled_img = labelImage(labels)
    outputImages(fileName, labeled_img)


if __name__ == "__main__":
    print("before main")
    main()