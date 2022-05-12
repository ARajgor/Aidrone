from djitellopy import Tello
from voice_reco import *
from speak_polly import *
import cv2
from multiprocessing import Process
import datetime

tello = Tello()
tello.connect()
# tello.streamoff()
# tello.streamon()
#
# frame_read =tello.get_frame_read()
count = 0

welcome_audio()


# def dt():
#     ct = format(datetime.datetime.now())
#     ct = ct.replace(':', '')
#     ct = ct.replace('.', '')
#     return ct
#
#
# def videoRecorder(frame_read):
#     start_time = time.time()
#     keepRecording = False
#     ct = dt()
#     while True:
#         frame_read = frame_read.frame
#
#         if keepRecording:
#             t = round(time.time() - start_time, 0)
#             minute = int(t / 60)
#             seconds = int(t % 60)
#             times = str(minute) + ':' + str(seconds)
#             cv2.putText(frame_read, times, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.putText(frame_read, "Recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         if not keepRecording:
#             start_time = time.time()
#             height, width, _ = frame_read.shape
#             video = cv2.VideoWriter(f'video{ct}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
#                                     (width, height))
#             keepRecording = True
#
#         if keepRecording:
#             video.write(frame_read)
#             time.sleep(1 / 60)
#
#
#     video.release()
#     return



def language_call():
    language("lang_ask")  # ask for language
    lang = ask_lang().lower()  # listen for language

    if lang == 'gujarati':  # language confirmation audio
        language("guj_set")
    elif lang == 'english':
        language("eng_set")

    return lang


lang = language_call()
# lang = 'english'
while True:
    string = takeCommand(lang).lower()
+
    if string == "change the language" or string == "લેંગ્વેજ બદલવી છે":
        lang = language_call()

    if string == "take off" or string == "શરૂ થા":
        print(string)
        movement_audio("take_off")
        tello.takeoff()
    if string == "land" or string == "લેન્ડ કર":
        print(string)
        movement_audio("landing")
        tello.land()

    if string == "take picture":
        key = 't'
        # cv2.imwrite(f"output/picture{count}.png", frame_read.frame)
        print("photo is taken")
        count = count + 1

    if string == "rotate left" or string == "ડાબી બાજુ":
        print(string)
        tello.rotate_counter_clockwise(90)

    if string == "rotate right" or string == "જમણી બાજુ":
        print(string)
        tello.rotate_clockwise(90)

    if string == "forward" or string == "આગળ આવ":
        print(string)
        # movement_audio("forward")
        tello.move_forward(30)

    if string == "back" or string == "પાછળ જા":
        print(string)
        # movement_audio("backward")
        tello.move_back(30)

    if string == "left" or string == "ડાબી સાઈડ" :
        print(string)
        tello.move_left(30)
    if string == "right" or string == "જમણી સાઈડ":
        print(string)
        tello.move_right(30)

    if string == "exit" or string == "ચાલો":
        tello.land()
        break

# tello.move_left(100)
# tello.rotate_clockwise(90)
# tello.move_forward(100)

