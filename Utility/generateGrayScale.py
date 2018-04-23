###################################################
##	generateGrayscale.py : utility script for converting color png image to greyscale using openCV
##	@author Youri Runghen-Vezina 2018
###################################################
import cv2
from os import listdir

SOURCE = 'SOURCE_DIR'
OUTDIR = 'DEST_DIR'

for f in listdir(SOURCE):
    if not f.endswith('.png'):
        continue
    print(f)
    img = cv2.imread(SOURCE+'/'+f)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(OUTDIR+'/'+f, grey)