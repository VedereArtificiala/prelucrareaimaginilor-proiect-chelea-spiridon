from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
#os pentru ca vom importa imagini dintr-un director personal

train_data_dir='/content/drive/MyDrive/Facial Recognition/data/train'
validation_data_dir='/content/drive/MyDrive/Facial Recognition/data/test/'

batch_size=64
img_size=(48,48)

generator_dateantrenare= ImageDataGenerator(
                    rotation_range=40,
                    rescale=1./255,
                    shear_range=0.4,
                    zoom_range=0.4,
                    horizontal_flip=True,
                    vertical_flip=False,
                    fill_mode='nearest')


generator_antrenare= generator_dateantrenare.flow_from_directory(
                    train_data_dir,
                    batch_size=batch_size,
                    class_mode='categorical',
                    target_size=img_size,
                    color_mode='grayscale',
                    shuffle=True)


validare_dateantrenare= ImageDataGenerator(rescale=1./255)


validare_antrenare= validare_dateantrenare.flow_from_directory(
                            validation_data_dir,
                            batch_size=batch_size,
                            class_mode='categorical',
                            target_size=img_size,
                            color_mode='grayscale',
                            shuffle=True)

emotii=['Happy','Neutral','Sad', 'Stressed', 'Stressed','Stressed','Surprise']

img, label= generator_antrenare.__next__()

model=Sequential() #unu dupa unu
nr_filtre = [16, 32, 64, 128, 256]
dim_img=(48, 48, 1)
dimensiu_filtru = (3, 3)
dim_max_pooling = (2, 2)
rata_dropout = [0.1, 0.1, 0.1, 0.1,0.1]   #se elimina aleatoriu niste neuroni
densitate = 512      #cu cat e mai mare, cu atat are cap mai multa de invatare

model.add(Conv2D(nr_filtre[0], kernel_size=dimensiu_filtru, activation='relu', input_shape=dim_img))


for i in range(1, len(nr_filtre)):
        model.add(Conv2D(nr_filtre[i], kernel_size=dimensiu_filtru, activation='relu'))
        model.add(MaxPooling2D(pool_size=dim_max_pooling))
        model.add(Dropout(rata_dropout[i - 1]))

 #flatten pune toate layerele intr-un vector
model.add(Flatten())
model.add(Dense(densitate,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(7,activation='softmax')) #asta e ultimu , output layer

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

ruta_antrenare = "/content/drive/MyDrive/Facial Recognition/data/train/"
ruta_test = "/content/drive/MyDrive/Facial Recognition/data/test"

numar_img_antr = 0
for ruta, dir, file in os.walk(ruta_antrenare):
    numar_img_antr =numar_img_antr+ len(file)

num_img_test = 0
for ruta, dir, file in os.walk(ruta_test):
    num_img_test =num_img_test+ len(file)

print(numar_img_antr)
print(num_img_test)

epochs=100

history=model.fit(generator_antrenare,
                steps_per_epoch=numar_img_antr//64,
                epochs=epochs,
                validation_data=validare_antrenare,
                validation_steps=num_img_test//64)

model.save('/content/drive/MyDrive/Facial Recognition/model_file_200epoch.h5')