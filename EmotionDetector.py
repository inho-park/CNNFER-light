import json

import cv2
import numpy as np
from keras.models import model_from_json
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import sys

# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#
# result = {'Angry': 0, 'Disgusted': 0, 'Fearful': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprised': 0}
#
# # load json and create model
# json_file = open('model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)
#
# # load weights into new model
# emotion_model.load_weights("model/emotion_model.h5")
# print("Loaded model from disk")
# # start the webcam feed
#
# # pass here your video path
# cap = cv2.VideoCapture("./video/test4.mp4")
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)
#
# flag = True
#
# out = cv2.VideoWriter('video/output.mp4', fourcc, fps, (w, h))
# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     # frame = cv2.resize(frame, 640, 480)
#     if not ret:
#         break
#     if ret:
#         face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#         # 머신 러닝을 회색 이미지로 구현했으므로 회색으로 이미지 변환
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # 감지된 얼굴 갯수 변수
#         # detect faces available on camera
#         num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#
#         # take each face available on the camera and Preprocess it
#         for (x, y, w, h) in num_faces:
#             if flag:
#                 flag = False
#                 # 얼굴에 직사각형 그림
#                 cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#                 # 얼굴 부분만 직사각형으로 잘라 회색 이미지로 변환
#                 roi_gray_frame = gray_frame[y:y + h, x:x + w]
#                 # 회색 이미지를 48 x 48 픽셀로 바꾸기 => 트레인에 사용한 사진 픽셀값
#                 cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#
#                 # 감정 예측 결과
#                 # predict the emotions
#                 emotion_prediction = emotion_model.predict(cropped_img)
#                 maxindex = int(np.argmax(emotion_prediction))
#                 # 원본 이미지
#                 # 7개의 감정 인덱스와 매핑한 변수
#                 # 분석한 값을 표현할 위치
#                 cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 result[emotion_dict[maxindex]] = result[emotion_dict[maxindex]] + 1
#             else:
#                 flag = True
#                 cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_AA)
#
#         cv2.imshow('Emotion Detection', frame)
#         out.write(frame)
#         # q 를 누르면 중단
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# for key in result:
#     print(result[key])
# print()
#
# file_path = 'emotion.json'
# with open(file_path, 'w') as f:
#     json.dump(result, f)
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
# clip = AudioFileClip('./video/test4.mp4')
# videoClip = VideoFileClip('./video/output.mp4')
# videoClip.audio = clip
# videoClip.write_videofile("./video/complete4.mp4")
# os.remove('./video/output.mp4')
# os.remove('./video/test.mp4')


def detecting_emotion(input_file_name):
    print("start detecting video")
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    result = {'Angry': 0, 'Disgusted': 0, 'Fearful': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprised': 0}

    # load json and create model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded model from disk")
    # start the webcam feed

    # pass here your video path
    cap = cv2.VideoCapture("./video/"+input_file_name+".mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    flag = True

    out = cv2.VideoWriter('video/output.mp4', fourcc, fps, (w, h))
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        # frame = cv2.resize(frame, 640, 480)
        if not ret:
            break
        if ret:
            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            # 머신 러닝을 회색 이미지로 구현했으므로 회색으로 이미지 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 감지된 얼굴 갯수 변수
            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                if flag:
                    flag = False
                    # 얼굴에 직사각형 그림
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                    # 얼굴 부분만 직사각형으로 잘라 회색 이미지로 변환
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    # 회색 이미지를 48 x 48 픽셀로 바꾸기 => 트레인에 사용한 사진 픽셀값
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    # 감정 예측 결과
                    # predict the emotions
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    # 원본 이미지
                    # 7개의 감정 인덱스와 매핑한 변수
                    # 분석한 값을 표현할 위치
                    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    result[emotion_dict[maxindex]] = result[emotion_dict[maxindex]] + 1
                else:
                    flag = True
                    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # cv2.imshow('Emotion Detection', frame)
            out.write(frame)
            # q 를 누르면 중단
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for key in result:
        print(result[key])
    print()

    file_path = 'emotion.json'
    with open(file_path, 'w') as f:
        json.dump(result, f)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    clip = AudioFileClip('./video/'+input_file_name+'.mp4')
    videoClip = VideoFileClip('./video/output.mp4')
    videoClip.audio = clip
    videoClip.write_videofile("./video/"+input_file_name+"_s.mp4")
    os.remove('./video/output.mp4')
    os.remove('./video/'+input_file_name+'.mp4')

def main(argv):
    detecting_emotion(argv[1])


if __name__ == "__main__":
    main(sys.argv)
