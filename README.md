# FER ( Facial Emotion Recognition )
## keras 를 이용하여 간단한 Emotion Recognition 기능 구현
***
1. 사용 전에 우선 data 에 fer 2013 dataset 파일 구하기
2. data/test, data/train 형태로 데이터들을 담기
3. model(학습한 내용 저장할 때 필요) 폴더와 video(영상 분석 후 저장할 때 필요) 폴더 만들기
4. 먼저 TrainEmotionDetector 파일을 이용하여 학습 먼저 시키기(너무 느리거나 컴퓨터의 부하가 심할 시 epochs 값 낮추기)
5. fed accuracy 평가를 하고 싶으면 EvaluateEmotionDetector 파일 실행
6. 영상을 분석하고 싶으면 TestEmotionDetector 에 사용할 영상 로컬 경로를 설정하여 실행(video 에 영상 저장)

***
### 1. 본인이 사용하고자 하는 훈련용 데이터들을 data folder 에 넣기
### 2. TrainEmotionDetector.py 를 통해 기계 학습 ( 3시간 정도 소요 )
### 3. TestEmotionDetector 에 원하는 영상 로컬 주소를 입력 후 실행
### 4. 해당 영상을 분석하여 영상에 표정을 실시간 텍스트로 출력
### 5. 4번의 텍스트까지 담아서 영상 저장