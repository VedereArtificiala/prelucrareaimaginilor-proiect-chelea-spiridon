import cv2
import numpy as np
import os
from keras.models import load_model
import face_recognition

model = load_model('model_file_200epoch.h5')
emotii = {0: 'Stress', 1: 'Stress', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Stress', 6: 'Surprise'}

folder_path = 'Poze/'
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(folder_path, filename))
        face_locations = face_recognition.face_locations(img)
        for (top, right, bottom, left) in face_locations:
            fata = img[top:bottom, left:right]
            conv_grayscale = cv2.cvtColor(fata, cv2.COLOR_BGR2GRAY)
            minimizare = cv2.resize(conv_grayscale, (48, 48))
            normalizare = minimizare / 255.
            mod_forma = np.reshape(normalizare, (1, 48, 48, 1))
            rez = model.predict(mod_forma)
            colt_stanga_sus=(left,top)
            colt_dreapta_jos=(right, bottom)
            cv2.rectangle(img, colt_stanga_sus, colt_dreapta_jos, (0, 0, 255), 3)
            cv2.putText(img, emotii[np.argmax(rez, axis=1)[0]], colt_stanga_sus, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow("Frame", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
