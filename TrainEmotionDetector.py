
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# 교육 모델에 컬러 이미지를 사용하면 데이터를 분석하는데 오랜 시간이 걸리므로 흑백 이미지 활용
# Preprocess all test images
# 학습용 이미지 자료 관련 설정
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
# 확인용 이미지 자료 관련 설정
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
emotion_model = Sequential()
# 회선 생성
# 버퍼 크기
# 커널 사이즈 설정
# 활성화 함수 (relu 는 Rectified Linear Unit 함수) 
# 사용할 사진 크기(px) 숫자 1 은 사진 = 흑백
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# 회선 추가
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# 과적합을 피하기 위한 풀 추가
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# 0 ~ 1 사이의 rate 를 기입하여 데이터 필터링
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# 모든 값을 평평하게함
emotion_model.add(Flatten())

# 밀집도 설정
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))

# 위에서 7 개의 분석 구조로 합이 1이 되게하는 함수 활성화 => softmax
emotion_model.add(Dense(7, activation='softmax'))
################################################################################

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])


# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        # 시스템이 느릴 시 epochs 를 낮추기
        epochs=50,
        # 유효성 검사
        validation_data=validation_generator,
        # 트레이닝에 대한 유효성 테스트를 위한 사진
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("model/emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('model/emotion_model.h5')

