# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:35:38 2021

@author: Asma Baccouche
"""

import cv2
from os import listdir

path1 = 'D:/Yufeng_Data/Masks_augmented/'
path2 = 'D:/Yufeng_Data/Current_augmented/'
path3 = 'D:/Yufeng_Data/Prior_augmented/'

files1 = listdir(path1)
files2 = listdir(path2)
files3 = listdir(path3)

inter = [elt for elt in files1 if elt in files2]
outer = [elt for elt in files2 if elt not in files1]


f = open("yufeng_annotation.txt", 'a')
for file in files2:
    if file in inter:
        img = cv2.imread(path1+file, 0)
        img2 = cv2.imread(path2+file)
        _,thresh = cv2.threshold(img,127,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        areas = [cv2.contourArea(c) for c in contours]
        a = sorted(areas,reverse=True)
        value = a[1]
        max_index = areas.index(value)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
       
        if 'MASS' in file:
            l = path2 + file + '\t' 
            l2 = str(x)+','+str(y)+','+str(x+w)+','+str(y+h)+',0' 
            l = l + l2 
        else:
            if 'CALC' in file or 'ARCH' in file:
                l = path2 + file + '\t' 
                l2 = str(x)+','+str(y)+','+str(x+w)+','+str(y+h)+',1' 
                l = l + l2
            else:
                l = path2 + file + '\t' 
                l2 = str(x)+','+str(y)+','+str(x+w)+','+str(y+h) 
                l = l + l2
    
        f.writelines("%s\n" % l)
    else:
        l = path2 + file + '\t' 
        f.writelines("%s\n" % l) 
f.close()

n = 'calc'
f = open("yufeng_"+n+"_annotation.txt", 'a')
for file in inter:
    img = cv2.imread(path1+file, 0)
    img2 = cv2.imread(path2+file)
    _,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    areas = [cv2.contourArea(c) for c in contours]
    a = sorted(areas,reverse=True)
    value = a[1]
    max_index = areas.index(value)
    cnt=contours[max_index]
    #l = path2 + file + '\t' 
    x,y,w,h = cv2.boundingRect(cnt)
   
    if n.upper() in file:
        l = path2 + file + '\t' 
        l2 = str(x)+','+str(y)+','+str(x+w)+','+str(y+h)+',0' 
        l = l + l2 
        f.writelines("%s\n" % l)
f.close()