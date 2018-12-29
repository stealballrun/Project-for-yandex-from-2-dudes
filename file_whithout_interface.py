#Imports
import numpy as np
import pandas as pd
import cv2
import keras
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Concatenate, Activation
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------





#Interface

#-------------------------------------------------------------------------------
#Post
#From_Interface:
#NAME
#process



#
f = open('name.txt', 'r')
k = f.read()
process = int(k[0])
NAME = k[2:]
f.close()
print(NAME)



#0(Функции)
def rle_decode(mask_rle, shape=(1280, 1918, 1)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    img = img.reshape(shape)
    return img


def keras_generator(gen_df, batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = cv2.imread('data/data/train/{}'.format(img_name))
            mask = rle_decode(mask_rle)
            
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            
            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)
        
        
        
#1
df = pd.read_csv('data/data/train_masks.csv')
train_df = df[:4000]
val_df = df[4000:]

for x, y in keras_generator(train_df, 16):
    break




if process==3:
    #---------------------------------------------------------------------------
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
    
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    
    
    
    #6(генератор обучения)
    batch_size = 16
    model.fit_generator(keras_generator(train_df, batch_size),
                  steps_per_epoch=32,
                  epochs=10,
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




if process == 2:
    #---------------------------------------------------------------------------
    #(Предсказание)
    #8(Загрузка нейросети)
    json_file = open("Model_author.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights_author.h5")
    
    loaded_model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    
    
    
    #9(Итог)
    #im_id = NAME #передается из интерфейса
    this_picture = cv2.imread(NAME) # в скобках должен быть NAME
    #this_picture = cv2.imread('pic01.jpg')
    this_picture = cv2.resize(this_picture, (256, 256))
    x[0] = this_picture
    pred = loaded_model.predict(x)
    this_mask = pred[0, ..., 0] > 0.5
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    axes[0].imshow(this_picture)
    axes[1].imshow(this_mask)
    plt.show()




if process == 1:
    #---------------------------------------------------------------------------
    #(Предсказание)
    #8(Загрузка нейросети)
    json_file = open("Model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights.h5")
    
    loaded_model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    
    
    
    #9(Итог)
    #im_id = NAME #передается из интерфейса
    this_picture = cv2.imread(NAME) # в скобках должен быть NAME
    #this_picture = cv2.imread('pic01.jpg')
    this_picture = cv2.resize(this_picture, (256, 256))
    x[0] = this_picture
    pred = loaded_model.predict(x)
    this_mask = pred[0, ..., 0] > 0.5
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    axes[0].imshow(this_picture)
    axes[1].imshow(this_mask)
    plt.show()  
