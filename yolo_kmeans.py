# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:24:12 2020

@author: Asma Baccouche
"""

import numpy as np


def iou(boxes, clusters):  # 1 box -> k clusters
    n = boxes.shape[0]
    k = cluster_number

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result

def avg_iou(boxes, clusters):
    accuracy = np.mean([np.max(iou(boxes, clusters), axis=1)])
    return accuracy

def kmeans(boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()
    clusters = boxes[np.random.choice(
        box_number, k, replace=False)]  # init k clusters
    while True:

        distances = 1 - iou(boxes, clusters)

        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            clusters[cluster] = dist(  # update clusters
                boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters

def result2txt(data):
    f = open(n+"_anchor.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()

def txt2boxes():
    f = open(filename, 'r')
    dataSet = []
    for line in f:
        if len(line.strip().split("\t")) > 1:
            infos = line.split("\t")[1]
            width = int(infos.split(",")[2]) - int(infos.split(",")[0])
            height = int(infos.split(",")[3]) - int(infos.split(",")[1])
            dataSet.append([width, height])
    result = np.array(dataSet)
    f.close()
    return result

def txt2clusters():
    all_boxes = txt2boxes()
    result = kmeans(all_boxes, k=cluster_number)
    result = result[np.lexsort(result.T[0, None])]
    result2txt(result)
    print("K anchors:\n {}".format(result))
    print("Accuracy: {:.2f}%".format(avg_iou(all_boxes, result) * 100))

cluster_number = 9
name = 'mass_and_non_mass'
n = "yufeng_"+name+"_annotation"
filename = n+".txt"
txt2clusters()