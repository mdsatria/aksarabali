import numpy as np
from PIL import Image
from pathlib import Path

PATH = "full/"
def readImgs(path, height=32, width=32, imgtype="*.jpg"):
    lst = np.empty([1, height, width, 3])
    img_file = []
    
    for file in Path(str(path)).rglob(imgtype):
        img_file.append(file)
        x = Image.open(file).resize(size=(height, width))
        x = np.array(x)
        x = x[np.newaxis, :, :, :]
        lst = np.concatenate((lst, x), axis=0)
    
    lst = np.delete(lst, 0, axis=0)
    
    return img_file, lst.astype(np.uint8)
 
temp = readImgs(PATH)
# np.save("data.npy", temp)

small_dt = np.loadtxt()
img = np.load("data.npy")

meanUint8 = [146.29538663869022, 122.42749753149018, 85.28546347939795]
meanFloat = [0.5737073985830996, 0.4801078334568246, 0.33445279795842325]
stdUint8 = [26.142449600739972, 25.98356495108353, 23.645138398823946]
stdFloat = [0.10251941019898031, 0.10189633314150415, 0.09272603293656458]
