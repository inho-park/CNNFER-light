import cv2
import numpy as np
from keras.models import model_from_json
from moviepy.editor import VideoFileClip, AudioFileClip
import os

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


result = [0, 0, 0, 0, 0, 0, 0]

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")
# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
cap = cv2.VideoCapture("./sample_video/emotion_sample9.webm")
audioclip = AudioFileClip("./sample_video/emotion_sample9.webm")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('video/output.mp4', fourcc, fps, (w, h))
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (int(w), int(h)))
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
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            result[maxindex] = result[maxindex] + 1
        cv2.imshow('Emotion Detection', frame)
        out.write(frame)
        # q 를 누르면 중단
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for i in range(0, 7):
    print(result[i], end=" ")
print()
cap.release()
out.release()
cv2.destroyAllWindows()


audioclip.write_audiofile('./audio/audio_sample.mp3')
videoClip = VideoFileClip('./video/output.mp4')
videoClip.audio = audioclip
videoClip.write_videofile("./video/complete.mp4")
os.remove('./audio/audio_sample.mp3')
os.remove('./video/output.mp4')
