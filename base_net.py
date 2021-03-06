from keras.layers import Input, Conv3D, Conv3DTranspose, Concatenate, MaxPool3D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from utils import generator9
import numpy as np
K.set_learning_phase(False)


# number of training scans
index_tr = np.array([0,1,2,4,5,6,7,8,9,10,11,12,14,15,16,17,
                     18,19,20,21,22,23,24,25,26,27,28,31,32,33])
index_va = np.array([35,36,37,38,39,40])

# Metrics
def dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# Losses
def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

# Model
K.clear_session()
I1 = Input(shape=(64, 64, 64, 4))
conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(I1)
conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3)
conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPool3D(pool_size=(2, 2, 2))(conv4)
conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
up6 = Concatenate(axis=-1)([up6, conv4])
conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
up7 = Concatenate(axis=-1)([up7, conv3])
conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
up8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
up8 = Concatenate(axis=-1)([up8, conv2])
conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
up9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
model = Model(inputs=[I1], outputs=conv10)

# Training
model.compile(optimizer=Adam(lr=1e-4),
              loss=dice_loss,
              metrics=[dice])

filepath = "/home/shas/projects/rrg-hamarneh/sponsored/shas/model1/weights-improvement-{epoch:02d}.hdf5"
my_checkpoint = ModelCheckpoint(filepath,
                                monitor='val_dice',
                                save_weights_only=True,
                                period=10,
                                verbose=1)

tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

training_generator = generator9(h5path='/dev/shm/shas/SFU_scaled_MICCAI', indices=index_tr, batchSize=15, imagesize=64, channel=4)
validation_generator = generator9(h5path='/dev/shm/shas/SFU_scaled_MICCAI', indices=index_va, batchSize=15, imagesize=64, channel=4)

model.fit_generator(generator=training_generator,
                    steps_per_epoch=300,
                    epochs=500,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=100,
                    use_multiprocessing=True,
                    callbacks=[my_checkpoint, tbCallBack])
