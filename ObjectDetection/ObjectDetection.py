import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread('images/1200px-Girl_and_cat.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = np.pad(img,((158,158),(0,0),(0,0)),mode='edge')
plt.imshow(img)

vgg = keras.applications.vgg16.VGG16(weights='imagenet')

def predict(img):
    im = cv2.resize(img,(224,224))
    im = keras.applications.vgg16.preprocess_input(im)
    pr = vgg.predict(np.expand_dims(im,axis=0))[0]
    return np.sum(pr[281:294]) # we know that VGG classes for cats are from 281 to 294

predict(img)


def predict_map(img,n):
    dx = img.shape[0] // n
    res = np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        for j in range(n):
            im = img[dx*i:dx*(i+1),dx*j:dx*(j+1)]
            r = predict(im)
            res[i,j] = r
    return res

fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[1].imshow(img)
ax[0].imshow(predict_map(img,10))


def generate_images(num_imgs, img_size=8, min_object_size = 1, max_object_size = 4):
    bboxes = np.zeros((num_imgs, 4))
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

    for i_img in range(num_imgs):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        imgs[i_img, x:x+w, y:y+h] = 1.  # set rectangle to 1
        bboxes[i_img] = [x, y, w, h]
    return imgs, bboxes

imgs, bboxes = generate_images(100000)
print(f"Images shape = {imgs.shape}")
print(f"BBoxes shape = {bboxes.shape}")


bb = bboxes/8.0
bb[0]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8,8)),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4)
])
model.compile('sgd','mse')
model.summary()

imgs_norm = (imgs-np.mean(imgs))/np.std(imgs)
model.fit(imgs_norm,bb,epochs=30)


def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I
    U = w1 * h1 + w2 * h2 - I
    return I / U


import matplotlib

test_imgs, test_bboxes = generate_images(500)
bb_res = model.predict((test_imgs-np.mean(imgs))/np.std(imgs))*8

plt.figure(figsize=(15,5))
for i in range(5):
    print(f"pred={bb_res[i]},act={test_bboxes[i]}, IOU={IOU(bb_res[i],test_bboxes[i])}")
    plt.subplot(1,5,i+1)
    plt.imshow(test_imgs[i])
    plt.gca().add_patch(matplotlib.patches.Rectangle((bb_res[i,1],bb_res[i,0]),bb_res[i,3],bb_res[i,2],ec='r'))
    #plt.annotate('IOU: {:.2f}'.format(IOU(bb_res[i],test_bboxes[i])),(bb_res[i,1],bb_res[i,0]+bb_res[i,3]),color='y')

np.array([IOU(a,b) for a,b in zip(test_bboxes,bb_res)]).mean()