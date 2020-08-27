from augment_and_mix import augment_and_mix
from PIL import Image
import numpy as np
import pandas as pd

np.random.seed(42)

def smallLabel(csvfile):
    """Create dataframe of class and file name

    """
    # read csv
    train_data = pd.read_csv(csvfile, delimiter=";", header=None)
    # drop columns
    train_data.drop(columns=[2], inplace=True)
    # find class with member less than 200
    lst = train_data[1].value_counts()
    lblLow = lst[(lst < 200) == True].index.tolist()
   
    result = [[]]
    for i in range (len(lblLow)):
        # find filename
        label = train_data[train_data[1]==lblLow[i]][0].values.tolist()
        for j in label:
            result.append([j, lblLow[i]])
    result.pop(0)
            
    return pd.DataFrame(result)


def readImgs(lstImg, height=32, width=32, imgtype="*.jpg", PATH="train/train_image/"):
    """Read collection of images in same class from list of image file names
        lstImg : file name of images in python list
        height : height of ouput image
        width : width of output image
        imgtype : image file extension 
    """
    lst = np.empty([1, height, width, 3])
    
    for i in lstImg:
        x = Image.open(PATH+i).resize(size=(height, width))
        x = np.array(x)
        x = x[np.newaxis, :, :, :]
        lst = np.concatenate((lst, x), axis=0)
    
    lst = np.delete(lst, 0, axis=0)
    return lst.astype(np.uint8)


def scalling(x, min_, max_):
    # return scalled array to 0-1
    return ((x - min_) / (max_ - min_))

def matScalling(l, min_, maks_):
    """ Scalling batch of image from float32 to uint8
        l is batch on image = [n, h, w, c]
        min_ is minimum value of l
        maks_ is maksimum value of l
    """  
    # declare size of batch
    n = l.shape[0]
    # vectorice
    img_scalled = l.ravel()
    # scalling with scalling function (0-1)
    img_scalled = np.apply_along_axis(scalling, 0, img_scalled, min_, maks_)
    # return batch of image into its original shape
    img_scalled = img_scalled.reshape(n,32,32,3)
    # return image value to 0-255
    img_scalled = img_scalled * 255
    return img_scalled.astype(np.uint8)

def classAugment(data, nMax, size):
    """ augment image with X class with Augmix
        return augmented image with number of size
        data : batch of image [n, h, w, c] in 4d numpy array uint8
        nMax : the number of image with class of X
        size : the desired number of augmented image to be generated
    """
    # convert batch of images to float32
    imgData = (data / 255.0).astype(np.float32)
    # declare variable to store output of function
    imgs = np.zeros([size, 32, 32, 3])
    # make random list
    lst = np.random.randint(nMax, size=size)
    for i in range(size):
        temp = augment_and_mix(imgData[lst[i], :, :, :])
        temp = temp[np.newaxis, :, :, :]
        imgs[i, :, :, :] = temp
    imgs = matScalling(imgs, imgs.min(), imgs.max())    
    return imgs


# read csv
label = smallLabel("train/gt_train.txt")
# create class label 
classLabel = label[1].unique()

csv_aug = [[]]

for i in range(len(classLabel)):
    # create list of image sources to be augmented
    lstImage = label[label[1] == classLabel[i]][0].tolist()
    # read image from lstImage and save it in 4d numpy array
    im = readImgs(lstImage)
    # find the number of source image
    nItem = im.shape[0]
    # find the number of augmented image to be generated
    nSize = 200 - im.shape[0]
    # augment image for nSize time and store it in 4d numpy array
    imA = classAugment(im, nItem, nSize)
    # looping to save augmented image
    for j in range(nSize):
        # convert augmented image in j-batch to PIL image
        saveIm = Image.fromarray(imA[j, :, :, :])
        # define directory of augmented image
        pathImAugment = "augmented_images/" 
        # define file name of augmented image
        fileName = ("{}_A_{}.jpg".format(classLabel[i], j+1))
        # save augmented image
        saveIm.save(pathImAugment+fileName)
        csv_aug.append([fileName, classLabel[i]])
        
    print("created {} augmented images in class {}".format(nSize, classLabel[i]))
        
csv_aug.pop(0)
pd.DataFrame(csv_aug).to_csv("augmented_data.csv", index=False, header=False)