import math

import cv2
import numpy as np
from object_detection import ObjectDetection

#initializing object detection

od = ObjectDetection()

cap = cv2.VideoCapture("./los_angeles.mp4")
#initialize count
count = 0
center_points_prev_frame=[]

scaling_factor = 0.5

tracking_objects ={}
track_id=1

while True:
    ret, frame = cap.read()
    count +=1
    if not ret:
        break
    center_points_cur_frame = []

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

    #Detect on frames
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x,y,w,h) = box
        cx = int((x + x +w) / 2)
        cy = int((y + y +h) / 2)
        center_points_cur_frame.append((cx,cy))
        print("frame no",count, x, y, w, h)
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)

    #bengining we compare prev and curr frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance =math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id+= 1

    else:

        tracking_objects_copy=tracking_objects.copy()
        center_points_cur_frame=center_points_cur_frame.copy()


        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                #update ids position
                if distance< 20:
                    tracking_objects[object_id]= pt
                    object_exists=True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                        continue

            #remove ids lost
            if not object_exists:
                tracking_objects.pop(object_id)

        #add new ids found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] =pt
            track_id +=1
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 3,(0,0,255), -1)
        cv2.putText(frame, str(object_id),(pt[0], pt[1]-7), 0, 1, (0,0,255), 2)


    print("tracking objects")
    print(tracking_objects)

    print("cur frame left points")
    print(center_points_cur_frame)





    # cv2.circle(frame, pt, 3,(0,0,255), -1)
    cv2.imshow("Frame ", frame)
    print("cur frame")
    print(center_points_cur_frame)
    print("prev frame")
    print(center_points_prev_frame)

    #make copy of points
    center_points_prev_frame= center_points_cur_frame.copy()
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()