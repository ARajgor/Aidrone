import cv2
from djitellopy import Tello
import numpy as np
import os

protoPath = os.path.join("model\\deploy.prototxt.txt")
modelPath = os.path.join("model\\res10_300x300_ssd_iter_140000.caffemodel")
S = 20
S2 = 5
UDOffset = 150
dimensions = (960, 720)
cWidth = int(dimensions[0]/2)
cHeight = int(dimensions[1]/2)
class FrontEnd(object):
    def __init__(self):
        self.tello = Tello()
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = True

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)
        self.tello.streamoff()
        self.tello.streamon()
        self.tello.takeoff()
        frames = self.tello.get_frame_read()


        print('Battery: ', self.tello.get_battery())

        while True:
            self.update()
            frame = frames.frame
            frame = cv2.resize(frame,dimensions)
            detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

            height, width = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
            detector.setInput(imageBlob)
            detections = detector.forward()

            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < 0.5:
                    continue
                else:
                    self.for_back_velocity = 0
                    self.left_right_velocity = 0
                    self.up_down_velocity = 0
                    self.yaw_velocity = 0

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                y_mid = int( (startY + endY) / 2) + UDOffset
                x_mid = int ((startX + endX) / 2)
                h = int(endY - startY)
                w = int(endX - startX)
                area = h * w

                vtr = np.array((cWidth, cHeight, 22000))
                vtg = np.array((x_mid, y_mid, area))
                vDistance = vtr-vtg

                print(f"startX:{startX}, startY:{startY}, endX:{endX}, endY:{endY}")
                print(f"y_mid:{y_mid}, x_mid:{x_mid}, area:{area}")
                print(f"vDistance[0]:{vDistance[0]},vDistance[1]:{vDistance[1]},vDistance[2]:{vDistance[2]}")

                if vDistance[0] < -100:
                    self.yaw_velocity = S
                    # self.left_right_velocity = S2
                elif vDistance[0] > 100:
                    self.yaw_velocity = -S
                    # self.left_right_velocity = -S2
                else:
                    self.yaw_velocity = 0

                    # for up & down
                if vDistance[1] > 55:
                    self.up_down_velocity = S
                elif vDistance[1] < -55:
                    self.up_down_velocity = -S
                else:
                    self.up_down_velocity = 0

                # F = 0
                # if abs(vDistance[2]) > 150:
                #     F = S

                # for forward back
                if 21000 < area < 23000:
                    self.for_back_velocity = 0
                elif area < 20000:
                    self.for_back_velocity = S
                elif area > 24000:
                    self.for_back_velocity = -S
                else:
                    self.for_back_velocity = 0


            cv2.imshow("Track",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.tello.land()
                break

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:

            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)
def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()