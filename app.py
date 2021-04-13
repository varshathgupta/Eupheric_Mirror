import streamlit as st
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from pydub import AudioSegment
from pydub.playback import play
import time


st.title("Eupheric Mirror")
st.image("pic1.jpeg")
t= "<h3 class = 'title blue'> Remove Your stress by interacting with us "
st.markdown(t,unsafe_allow_html = True)

def about():
    st.title("About:")
    st.write(
        """
        Hello all these our small prototype of our Euperic Mirror 

        By this app one can remove their stress by interacting with this software. 

        We are doing this app for our protyping submission.

        """
    )

activities = ["Home", "About"]
choice = st.sidebar.selectbox("Choose",activities)

if choice == "Home":
    st.write("""
    Are you feel stressed, then don't worry.

    Your stress is totally removed now.

    Click below button to start analysing you.
    """)
    if st.button("Start"):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # command line argument
        ap = argparse.ArgumentParser()
        ap.add_argument("--mode",help="train/display")
        mode = ap.parse_args().mode

        # plots accuracy and loss curves
        def plot_model_history(model_history):

            #Plot Accuracy and Loss curves given the model_history

            fig, axs = plt.subplots(1,2,figsize=(15,5))
            # summarize history for accuracy
            axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
            axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
            axs[0].set_title('Model Accuracy')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_xlabel('Epoch')
            axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
            axs[0].legend(['train', 'val'], loc='best')
            # summarize history for loss
            axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
            axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
            axs[1].set_title('Model Loss')
            axs[1].set_ylabel('Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
            axs[1].legend(['train', 'val'], loc='best')
            fig.savefig('plot.png')
            st.write(plt.show())


        train_dir = './Datasets/train'
        val_dir = './Datasets/validation'

        num_train = 28709
        num_val = 7178
        batch_size = 64
        num_epoch = 70

        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(48,48),
                batch_size=batch_size,
                color_mode="grayscale",
                class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(48,48),
                batch_size=batch_size,
                color_mode="grayscale",
                class_mode='categorical')

        # Create the model
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        # If you want to train the same model or try other models, go for this
        if mode == "train":
            model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
            model_info = model.fit_generator(
                    train_generator,
                    steps_per_epoch=num_train // batch_size,
                    epochs=num_epoch,
                    validation_data=validation_generator,
                    validation_steps=num_val // batch_size)
            plot_model_history(model_info)
            model.save_weights('model.h5')
            

        # emotions will be displayed on your face from the webcam feed
        elif mode == "display":
            model.load_weights('model.h5')

            # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

            # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",4: "Neutral", 5: "Sad", 6: "Surprised"}

            # start the webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                st.write(cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2))
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                st.write( cv2.imshow('Video', cv2.resize(frame,(1080,720),interpolation = cv2.INTER_CUBIC)))

                break
            


            if maxindex == 0:
                st.write(cv2.putText(frame, "you are angry", (x  , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA))
                a = AudioSegment.from_mp3("./Audio/Angry/1.mp3")
                b = AudioSegment.from_mp3("./Audio/Angry/2.mp3")
                louder_song = a+8
                play(louder_song)
                louder_song = b+8
                play(louder_song)
                exit()

            elif maxindex == 1:
                st.write(cv2.putText(frame, "you are disgusted", (x , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA))
                a = AudioSegment.from_mp3("./Audio/disgusted.mp3")
                louder_song = a+6
                play(louder_song)
                exit()

            elif maxindex == 2:
                
                st.write( cv2.putText(frame, "you are afraid", (x , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA))
                a = AudioSegment.from_mp3("./Audio/fear.mp3")
                louder_song = a+6
                play(louder_song)
                exit()

            elif maxindex == 3:
                
                st.write(cv2.putText(frame, "you are happy", (x , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA))
                a = AudioSegment.from_mp3("./Audio/Happy/1.mp3")
                b = AudioSegment.from_mp3("./Audio/Happy/2.mp3")
                louder_song = a+6
                play(louder_song)
                louder_song = b+6
                play(louder_song)
                exit()
            
            elif maxindex == 4:
                st.write(cv2.putText(frame, "you are neutral", (x , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA))
                exit()
                
            elif maxindex == 5:
                
                st.write(cv2.putText(frame, "you are sad", (x , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA))
                a = AudioSegment.from_mp3("./Audio/Sad/1.mp3")
                b = AudioSegment.from_mp3("./Audio/Sad/2.mp3")
                c = AudioSegment.from_mp3("./Audio/Sad/4.mp3")
                louder_song = a + 8
                play(louder_song)
                time.sleep(1)
                louder_song = b + 8
                play(louder_song)
                time.sleep(2)
                louder_song = c+ 8
                play(louder_song)
                exit()
            else:
            
                st.write(cv2.putText(frame, "you are surprised", (x , y + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA))
                a = AudioSegment.from_mp3("./Audio/surprise.mp3")
                louder_song = a+6
                play(louder_song)
                exit()
            st.write(cv2.imshow('Video', cv2.resize(frame,(1080,720),interpolation = cv2.INTER_CUBIC)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

elif choice == "About":
    about()


