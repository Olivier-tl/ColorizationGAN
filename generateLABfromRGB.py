import cv2
from os import listdir

SOURCE = '/home/aksii/INF8225/GanProject/TRAIN/color'
OUTDIR = '/home/aksii/INF8225/GanProject/TRAIN/color_lab'

for f in listdir(SOURCE):
    if not f.endswith('.png'):
        continue
    print(f)
    img = cv2.imread(SOURCE+'/'+f)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    cv2.imwrite(OUTDIR+'/'+f, grey)