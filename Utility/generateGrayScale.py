import cv2
from os import listdir

SOURCE = '/home/aksii/INF8225/GanProject/TRAIN/IMG_96_160'
OUTDIR = '/home/aksii/INF8225/GanProject/TRAIN/GRAY'

for f in listdir(SOURCE):
    if not f.endswith('.png'):
        continue
    print(f)
    img = cv2.imread(SOURCE+'/'+f)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(OUTDIR+'/'+f, grey)