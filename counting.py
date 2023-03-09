
class tracker_counter:
    def __init__(self, object, image, filename):
        self.Enter_frame = []
        self.Exit_frame = []
        self.Fish_count = 0
        self.line = [image.shape[1] // 2, image.shape[0] // 2] # [x,y] = [width, height] = [col, row]
        self.line_as_box = [(image.shape[1] // 2), 0, (image.shape[1] // 2) + 1, image.shape[0]] # [x_top, y_top, x_bottom, y_bottom]
        self.tracker_object = object
        self.bbox = self.tracker_object.to_tlbr()
        self.bbox_front = [self.bbox[0], self.bbox[1] // 2] # [x, y]
        self.bbox_back = [self.bbox[2], self.bbox[3] // 2] # [x, y]
        self.bbox_center = [self.bbox[0] + (self.bbox[2] - self.bbox[0]) // 2, self.bbox[1] + (self.bbox[3] - self.bbox[1]) // 2] # [x, y]
        self.old_center_side = None
        self.middle_start = None
        self.state = 0 # 0: waiting for enter frame, 1: waiting for exit frame
        self.file_name = filename
        self.fish_count_frames = []

    def update_bbox(self):
        self.bbox = self.tracker_object.to_tlbr()
        self.bbox_front = [self.bbox[0], self.bbox[1] // 2]
        self.bbox_back = [self.bbox[2], self.bbox[3] // 2]
        self.bbox_center = [self.bbox[0] + (self.bbox[2] - self.bbox[0]) // 2,
                            self.bbox[1] + (self.bbox[3] - self.bbox[1]) // 2]  # [x, y]


    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def count(self, frame_id):
        # First, update the bbox information
        self.update_bbox()
        # Second, check if the salmon started in the middle of the line.
        if self.state == 0:
            if self.middle_start is None and self.bb_intersection_over_union(self.bbox, self.line_as_box) > 0:
                self.middle_start = True
                self.Enter_frame.append(frame_id)
                self.state = 1
            elif self.line[0] > self.bbox_center[0] and self.bb_intersection_over_union(self.bbox, self.line_as_box) == 0: # If not, then we find which side of the line the salmon is on.
                self.old_center_side = "left"
                self.state = 1
            elif self.line[0] < self.bbox_center[0] and self.bb_intersection_over_union(self.bbox, self.line_as_box) == 0:
                self.old_center_side = "right"
                self.state = 1
        elif self.state == 1: # Third, check if the salmon bbox intersects with the line.
            if self.bb_intersection_over_union(self.bbox, self.line_as_box) > 0:
                self.state = 2
        elif self.state == 2: # Now we wait until the salmon is completely out of the line.
            if self.bb_intersection_over_union(self.bbox, self.line_as_box) == 0:
                # We then check which side of the line the salmon is on.
                if self.line[0] > self.bbox_center[0]: # Left side
                    if self.middle_start == True or self.old_center_side == "right":
                        self.old_center_side = "left"
                        self.Exit_frame.append(frame_id)
                        self.Fish_count += 1
                        self.middle_start = False
                        self.state = 0
                        self.fish_count_frames.append(1)
                    else:
                        self.fish_count_frames.append(0)
                        self.middle_start = False
                        self.state = 0
                        self.Exit_frame.append(frame_id)
                elif self.line[0] < self.bbox_center[0]: # Right side
                    if self.middle_start == True or self.old_center_side == "left":
                        self.old_center_side = "right"
                        self.Exit_frame.append(frame_id)
                        self.Fish_count -= 1
                        self.state = 0
                        self.middle_start = False
                        self.fish_count_frames.append(-1)
                    else:
                        self.fish_count_frames.append(0)
                        self.Exit_frame.append(frame_id)
                        self.state = 0
                        self.middle_start = False












