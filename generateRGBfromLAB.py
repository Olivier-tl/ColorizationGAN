import cv2
from os import listdir

SOURCE = '/home/aksii/INF8225/INF8225_TP4/'
OUTDIR = '/home/aksii/INF8225/GanProject/RESULT'

for f in listdir(SOURCE):
    if not f.endswith('.png'):
        continue
    print(f)
    img = cv2.imread(SOURCE+'/'+f)
    grey = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    cv2.imwrite(OUTDIR+'/'+f, grey)