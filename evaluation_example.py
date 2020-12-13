from keras.layers import Input, Conv3D, Conv3DTranspose, Concatenate, MaxPool3D, Flatten, Dense
from keras.models import Model
from keras import backend as K
from utils import generator
import numpy as np
K.set_learning_phase(False)

# number of training scans
scans = np.arange(0, 44, 1)
index_tr = np.random.choice(scans.shape[0], 33, replace=False)
index_te = np.setxor1d(scans, index_tr)

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
falt = Flatten()(conv3)
fc1 = Dense(100, activation='relu')(falt)
fc2 = Dense(56, activation='softmax')(fc1)
up6 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3)
up6 = Concatenate(axis=-1)([up6, conv2])
conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)
conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)
up7 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
up7 = Concatenate(axis=-1)([up7, conv1])
conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)
conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
model = Model(inputs=[I1], outputs=[conv10, fc2])

model.load_weights(filepath='/local-scratch/shahab_aslani/Test/Code/model/weights-improvement-01.hdf5')

training_generator = generator(h5path='data.hdf5', indices=index_te, batchSize=4, is_train=False, imagesize=64,
                               channel=4, number_class=56)

seg, cl = model.predict(training_generator[0])