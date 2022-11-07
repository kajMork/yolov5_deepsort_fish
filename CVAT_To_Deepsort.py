import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import os
import math
import shutil
import json
import subprocess
import cv2 as cv
def load_data():
    folder = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\vcat_model\annotations.xml"
    tree = ET.parse(folder)
    root = tree.getroot()
    return root

def find_cropped_frame(frame_nr, images, xtl, ytl, xbr, ybr):
    image = cv2.imread(images[int(frame_nr)])
    y_region = slice(ytl, ybr, 1)
    x_region = slice(xtl, xbr, 1)
    image_crop = image[y_region, x_region]
    return image_crop


if __name__ == "__main__":
    mode = "yolo"
    images = glob.glob(r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\vcat_model\images\default\*.png")
    yolo_images = glob.glob(r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\default\*.png")
    yolo_annotations = glob.glob(r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\default\*.txt")
    print(yolo_images[0], yolo_images[3885])
    with open(r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\test_train_val_split.json") as f:
        test_train_val_split = json.load(f)

    test_videos = test_train_val_split['test']
    train_videos = test_train_val_split['train']
    val_videos = test_train_val_split['val']

    root = load_data()
    camera = "C1"
    root_path = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\vcat_model\images\split"
    if mode == "deepsort":
        for child in root:
            if child.tag == "track":
                #print(child.tag, child.attrib)
                track_id = child.get("id")
                task_id = child.get("task_id")
                destination_path = root_path + "\\" + track_id+ "_" + task_id
                for subchild in child:
                    #print(subchild.tag, subchild.attrib)
                    frame_nr = subchild.get('frame')
                    xtl = math.floor(float(subchild.get('xtl')))
                    ytl = math.floor(float(subchild.get('ytl')))
                    xbr = math.ceil(float(subchild.get('xbr')))
                    ybr = math.ceil(float(subchild.get('ybr')))
                    img_crop = find_cropped_frame(frame_nr, images, xtl, ytl, xbr, ybr)
                    track_id = track_id.zfill(4)
                    tracklet = "T"+track_id.zfill(4)
                    frame_count ="F"+frame_nr.zfill(4)
                    filename = track_id+camera+tracklet+frame_count+".jpg"
                    ret = cv2.imwrite(destination_path+"\\"+filename, img_crop)
                    if ret == False:
                        #print("Error writing file")
                        os.mkdir(destination_path)
                        ret = cv2.imwrite(destination_path + "\\" + filename, img_crop)
                    #cv2.imshow('img', img_crop)
                    #cv2.waitKey(0)

    #Yolov5 format split
    start_point = 0
    end_point = 1
    if mode == "yolo":
        for child in root:
            if child.tag == "meta":
                for subchild in child:
                    if subchild.tag == "project":
                        for subsubchild in subchild:
                            if subsubchild.tag == "tasks":
                                for subsubsubchild in subsubchild:
                                    if subsubsubchild.tag == "task":
                                        video_id = subsubsubchild[0].text #video id
                                        video_name = subsubsubchild[1].text #video name
                                        start_frame = subsubsubchild[9].text #start frame
                                        stop_frame = subsubsubchild[10].text #stop frame
                                        min_range = int(start_frame)+start_point
                                        max_range = int(stop_frame)+end_point
                                        image_split = yolo_images[min_range: max_range]
                                        annotaion_split = yolo_annotations[min_range: max_range]
                                        start_point = start_point + int(stop_frame) +1
                                        end_point = end_point + int(stop_frame) +1
                                        if video_name in test_videos["file_name"]:
                                            save_path_img = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\split\images" + "\\test"
                                            save_path_ann = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\split\labels" + "\\test"
                                        elif video_name in train_videos["file_name"]:
                                            save_path_img = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\split\images" + "\\train"
                                            save_path_ann = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\split\labels" + "\\train"
                                        elif video_name in val_videos["file_name"]:
                                            save_path_img = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\split\images" + "\\val"
                                            save_path_ann = r"D:\OneDrive\AVS\7.Semester\Image Processing and Computer Vision\yolo_model\split\labels" + "\\val"
                                        for anno in annotaion_split:
                                            shutil.copy(anno, save_path_ann)
                                        for image in image_split:
                                            cv.imread(image)
                                            cv.imwrite(save_path_img+"\\"+image.split("\\")[-1], cv.imread(image))
                                        print("Done with video: ", video_name)
                                        print("for range: ", min_range, ", ", max_range)
                                        print("saved to: ", save_path_img)




"""
In bbox "0065C1T0002F0016.jpg", "0065" is the ID of the pedestrian. "C1" denotes the first camera (there are totally 6 cameras).
"T0002" means the 2th tracklet. "F016" is the 16th frame within this tracklet. For the tracklets, their names are accumulated for each ID; but for frames, they start from "F001" in each tracklet.
"""