import cv2 as cv




def main():
    video_import_path = "../../datasets/fish_test/fish_video8.mp4"
    img_export_path = "../../yolov5/deep_sort/FISH/train/FISH01/img1/"

    # Load the video
    vidcap = cv.VideoCapture(video_import_path)

    # Save each frame as an image
    count = 1
    while True:
        success, image = vidcap.read()
        if not success:
            break
        print('Read a new frame: ', success)
        cv.imwrite(img_export_path + "%06d.jpg" % count, image)
        count += 1

if __name__ == "__main__":
    main()