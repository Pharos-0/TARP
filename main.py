import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO
import serial
from time import sleep
def detected():
    data=serial.Serial("COM7",9600,timeout=.1)
    data.write('a'.encode('utf-8'))
    sleep(1)
    print("Data 1")
    def detected2():
        data=serial.Serial("COM7",9600,timeout=.1)
        data.write('b'.encode('utf-8'))
        sleep(1)
        print("Data 2")
        def detected3():
            data=serial.Serial("COM7",9600,timeout=.1)
            data.write('c'.encode('utf-8'))
            sleep(1)
            print("Data 3")
            def detected4():
                data=serial.Serial("COM7",9600,timeout=.1)
                data.write('d'.encode('utf-8'))
                sleep(1)
                print("Data 4")
                def detected5():
                    data=serial.Serial("COM7",9600,timeout=.1)
                    data.write('e'.encode('utf-8'))
                    sleep(1)
                    print("Data 5")
hand_detection_method = 'yolo'
if hand_detection_method == 'solo':
    hand = SOLO(weights='weights/solo.h5', threshold=0.8)
    if hand_detection_method == 'yolo':
        hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
        else:
            assert False, "'" + hand_detection_method + \
                  "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"
fingertips = Fingertips(weights='weights/fingertip.h5')
cam = cv2.VideoCapture(0)
print('Unified Gesture & Fingertips Detection')
while True:
    ret, image = cam.read()
    if ret is False:
        break
    tl, br = hand.detect(image=image)
    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        finger_count = int(np.sum(prob)) 
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        cv2.putText(image, f'Finger Count: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(finger_count)
        if finger_count==1:
            detected()
        if finger_count==2:
            detected2()
        if finger_count==3:
            detected3()
        if finger_count==4:
            detected4()
        if finger_count==5:
            detected5()
    if cv2.waitKey(1) & 0xff == 27:
        finger_count=str(finger_count)
        break
    # display image
    cv2.imshow('Unified Gesture & Fingertips Detection', image)
cam.release()
cv2.destroyAllWindows()
