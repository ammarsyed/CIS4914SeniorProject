from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys

def computeDerivative(img, sigmaX, sigmaY):
    img = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigmaX, sigmaY=sigmaY)
    dxKernel = np.array([[1],[0],[-1]])
    dyKernel = np.array([[1,0,-1]])
    dxxKernel = np.array([[1],[-2],[1]])
    dyyKernel = np.array([[1,-2,1]])
    dxyKernel = np.array([[1,-1],[-1,1]])

    dx = cv2.filter2D(img,-1, dxKernel)
    dy = cv2.filter2D(img,-1, dyKernel)
    dxx = cv2.filter2D(img,-1, dxxKernel)
    dyy = cv2.filter2D(img,-1, dyyKernel)
    dxy = cv2.filter2D(img,-1, dxyKernel)
    return dx, dy, dxx, dyy, dxy

def computeMagnitude(dxx, dyy):
    dxx = dxx.astype(float)
    dyy = dyy.astype(float)
    mag = cv2.magnitude(dxx, dyy)
    phase = mag*180./np.pi
    return mag, phase

def computeHessian(dx, dy, dxx, dyy, dxy):
    pt=[]
    dir=[]
    val=[]
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if dxy[y,x] > 0:
                hes = np.zeros((2,2))
                hes[0,0] = dxx[y,x]
                hes[0,1] = dxy[y,x]
                hes[1,0] = dxy[y,x]
                hes[1,1] = dyy[y,x]
                ret, eigVal, eigVect = cv2.eigen(hes)
                if np.abs(eigVal[0,0]) >= np.abs(eigVal[1,0]):
                    nx = eigVect[0,0]
                    ny = eigVect[0,1]
                else:
                    nx = eigVect[1,0]
                    ny = eigVect[1,1]
                den = dxx[y,x]*nx*nx + dyy[y,x]*ny*ny + 2*dxy[y,x]*nx*ny
                if den != 0:
                    T = -(dx[y,x]*nx + dy[y,x]*ny)/den
                    if np.abs(T*nx) <= 0.5 and np.abs(T*ny) <= 0.5:
                        pt.append((x,y))
                        dir.append((nx,ny))
                        val.append(np.abs(dxy[y,x]+dxy[y,x]))
    return pt, dir, val


if len(sys.argv) == 1:
    print("requires image Source as an argument")
    sys.exit()

img = cv2.imread(sys.argv[1])
if img is None:
    print("image filepath is invalid or doesn't exist")
    sys.exit()

# Initialization
img = cv2.bitwise_not(img)
blur = cv2.GaussianBlur(img, (5, 5), 0)
thinned = cv2.ximgproc.thinning(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
img = thinned
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,480))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Curve Detection
dx, dy, dxx, dyy, dxy = computeDerivative(gray_img, 1.1, 1.1)
normal, phase = computeMagnitude(dxx, dyy)

pt, dir, val = computeHessian(dx, dy, dxx, dyy, dxy)

idx = np.argsort(val)
idx = idx[::-1][:10000]
for i in range(0, len(idx)):
    img = cv2.circle(img, (pt[idx[i]][0], pt[idx[i]][1]), 1, (255, 0, 0), 1)

# Connected Components
normal = np.array(normal, dtype=np.uint8)
normal = cv2.threshold(normal, 127, 255, cv2.THRESH_BINARY)[1]
num_labels, labels = cv2.connectedComponents(normal)

# Conversion
cv2.imwrite('Test.png', phase)
test = cv2.imread('Test.png')

blur = cv2.GaussianBlur(test, (5, 5), 0)
test = cv2.ximgproc.thinning(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

test = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)[1]
num_labels, labels = cv2.connectedComponents(test)

# Coloring
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

# Image after Component Labeling
plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Component Labeling Image")
plt.show()
