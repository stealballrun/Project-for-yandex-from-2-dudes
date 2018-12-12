import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Activation
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D


#From_Interface:
#NAME

#0(Функции)
def rle_decode(mask_rle, shape=(1280, 1918, 1)):
    img1 = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    img2 = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    img3 = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths    
    for lo, hi in zip(starts, ends):
        img1[lo:hi] = 1
        img2[lo:hi] = 1
        img3[lo:hi] = 1
        
    img = np.array([img1,img2,img3])
    return Image.fromarray(img)


def keras_generator(gen_df, batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            mask = rle_decode(mask_rle)
            img = Image.open('C:/Users/Ivan/Documents/Yandex_1/data/data/train/{}'.format(img_name))
            img.resize((256, 256), Image.NEAREST)
            mask.resize((256, 256), Image.NEAREST)
            
            mask = np.asarray(mask)[0]
            img = np.asarray(img)
            
            x_batch += [img]
            y_batch += [mask]
        
        
        y_batch = np.array(y_batch)
        x_batch = np.array(x_batch) / 255.

        yield x_batch, np.expand_dims(y_batch, -1)
        
        
#1
df = pd.read_csv('C:/Users/Ivan/Documents/Yandex_1/data/data/train_masks.csv')
train_df = df[:4000]
val_df = df[4000:]

img_name, mask_rle = train_df.iloc[4]
img = Image.open('C:/Users/Ivan/Documents/Yandex_1/data/data/train/{}'.format(img_name))
img = img.resize((256, 256), Image.NEAREST)
mask = rle_decode(mask_rle)

for x, y in keras_generator(train_df, 16):
    break


#Модель
#2(Вход и обработка до малогого разрешения)
inp = Input(shape=(256, 256, 3))

conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
conv_1_1 = Activation('relu')(conv_1_1)

conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
conv_1_2 = Activation('relu')(conv_1_2)

pool_1 = MaxPooling2D(2)(conv_1_2)


conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
conv_2_1 = Activation('relu')(conv_2_1)

conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
conv_2_2 = Activation('relu')(conv_2_2)

pool_2 = MaxPooling2D(2)(conv_2_2)


conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
conv_3_1 = Activation('relu')(conv_3_1)

conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
conv_3_2 = Activation('relu')(conv_3_2)

pool_3 = MaxPooling2D(2)(conv_3_2)


conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
conv_4_1 = Activation('relu')(conv_4_1)

conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
conv_4_2 = Activation('relu')(conv_4_2)

pool_4 = MaxPooling2D(2)(conv_4_2)


#3(расширение)
up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4)
conc_1 = Concatenate()([conv_4_2, up_1])

conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1)
conv_up_1_1 = Activation('relu')(conv_up_1_1)

conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
conv_up_1_2 = Activation('relu')(conv_up_1_2)


up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
conc_2 = Concatenate()([conv_3_2, up_2])

conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(conc_2)
conv_up_2_1 = Activation('relu')(conv_up_2_1)

conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
conv_up_2_2 = Activation('relu')(conv_up_2_2)


up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
conc_3 = Concatenate()([conv_2_2, up_3])

conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(conc_3)
conv_up_3_1 = Activation('relu')(conv_up_3_1)

conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
conv_up_3_2 = Activation('relu')(conv_up_3_2)



up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
conc_4 = Concatenate()([conv_1_2, up_4])
conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(conc_4)
conv_up_4_1 = Activation('relu')(conv_up_4_1)

conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1)


#4(готовая модель)
result = Activation('sigmoid')(conv_up_4_2)
model = Model(inputs=inp, outputs=result)

model_json = model.to_json()
json_file = open("Model.json", "w")
json_file.write(model_json)
json_file.close()


#5(веса в нейросети)
best_w = keras.callbacks.ModelCheckpoint('w_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

last_w = keras.callbacks.ModelCheckpoint('W_last.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=True,
                                mode='auto',
                                period=1)


callbacks = [best_w, last_w]


adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(adam, 'binary_crossentropy')


#6(генератор обучения)
batch_size = 16
model.fit_generator(keras_generator(train_df, batch_size),
              steps_per_epoch=100,
              epochs=30,
              verbose=1,
              callbacks=callbacks,
              validation_data=keras_generator(val_df, batch_size),
              validation_steps=50,
              class_weight=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              shuffle=True,
              initial_epoch=0)


#7(сохранение нейросети)
model.save_weights("weights.h5")
#Обучение модели закончено:
    # Модель: 'Model.json'
    # Веса: 'weights.h5'


#8(Загрузка нейросети)
json_file = open("C:/Users/Ivan/Documents/Yandex_1/Model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("C:/Users/Ivan/Documents/Yandex_1/weights.h5")

loaded_model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])


#9(Итог)
#im_id = NAME #передается из интерфейса

this_picture = Image.open('c7a94c46a3b2_07.jpg') # в скобках должен быть NAME
this_picture.save('picture.jpg')
this_picture = this_picture.resize((256, 256), Image.NEAREST)

this_mask = loaded_model.predict(np.expand_dims(this_picture,0)[0, ..., 0] > 0.5)

this_mask.save('mask.jpg')

#В интерфейс:
# Picture: 'picture.jpg'
# Mask: 'mask.jpg'