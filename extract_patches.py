import numpy as np
import nibabel as nib
import os
from utils import patch_gen, image_gen, concat_patch, selective_patch, one_hot_vector
import h5py


# Read images
Input_path = '/local-scratch/shahab_aslani/Test/Data/Processed/All/'
files_images = os.listdir(Input_path)

# Separate files
Flair = [i for i, x in enumerate(files_images) if x[24] == "F"]
Mprage = [i for i, x in enumerate(files_images) if x[24] == "P"]
T2 = [i for i, x in enumerate(files_images) if x[24] == "L"]
PD = [i for i, x in enumerate(files_images) if x[24] == "S"]
Mask = [i for i, x in enumerate(files_images) if x[24] == "M"]

Flair_list = [None] * len(Flair)
Mprage_list = [None] * len(Mprage)
T2_list = [None] * len(T2)
PD_list = [None] * len(PD)
Mask_list = [None] * len(Mask)

i = 0;
j = 0;
k = 0;
l = 0;
m = 0

for name in files_images:

    if name[24] == 'F':
        Flair_list[i] = name
        i = i + 1

    if name[24] == 'P':
        Mprage_list[j] = name
        j = j + 1

    if name[24] == 'L':
        T2_list[k] = name
        k = k + 1

    if name[24] == 'S':
        PD_list[l] = name
        l = l + 1

    if name[24] == 'M':
        Mask_list[m] = name
        m = m + 1

Flair_list = sorted(Flair_list)
Mprage_list = sorted(Mprage_list)
T2_list = sorted(T2_list)
PD_list = sorted(PD_list)
Mask_list = sorted(Mask_list)
Scanner_list = np.array([], dtype=int)
counter = 0

# get one hot classification network for all data
train_Y_one_hot = one_hot_vector(Mprage_list)

with h5py.File('data', 'w') as f:
    for i in range(len(Mprage_list)):
        # read input images
        # Flair
        file = nib.load(Input_path + Flair_list[i])
        image_ori_FL = file.get_data()

        # T1
        file = nib.load(Input_path + Mprage_list[i])
        image_ori_MP = file.get_data()

        # T2
        file = nib.load(Input_path + T2_list[i])
        image_ori_T2 = file.get_data()

        # PD
        file = nib.load(Input_path + PD_list[i])
        image_ori_PD = file.get_data()

        # MASK
        file = nib.load(Input_path + Mask_list[i])
        image_ori_MA = file.get_data()

        # generate patches from images
        patches1 = patch_gen(image_ori_FL, (64, 64, 64))  # Flair
        patches2 = patch_gen(image_ori_MP, (64, 64, 64))  # T1
        patches3 = patch_gen(image_ori_T2, (64, 64, 64))  # T2
        patches4 = patch_gen(image_ori_PD, (64, 64, 64))  # PD
        patches5 = patch_gen(image_ori_MA, (64, 64, 64))  # Mask
        patches5 = patches5[..., np.newaxis]  # convert mask file same as other files

        # concatenate all patches from modalities
        patches_new = concat_patch(patches1, patches2, patches3, patches4)

        # select patches with lesions
        labels, patches = selective_patch(patches_new, patches5)

        # select correct one-hot classification label for set
        labels_cl = train_Y_one_hot[counter: counter + patches.shape[0]]
        counter = counter + 150

        # save file in hdf5
        f['data%s' % i] = patches
        f['label%s_se' % i] = labels
        f['label%s_cl' % i] = labels_cl
        
        print(i)



    # with open('./trainMS_list.txt', 'a') as f:
    #     f.write('data.h5\n' % i)

    # generate image from patches
    #image = image_gen(patches1, (64, 64, 64), (182, 218, 182))
    # # save generated image
    # file._data_cache = image
    # nib.save(file, os.path.join('test.nii.gz'))


    # # save files
    # np.save('/local-scratch/shahab_aslani/Test/Code/data/patch_images_train_SFU.npy', patches_new)
    # np.save('/local-scratch/shahab_aslani/Test/Code/data/patch_mask_train_SFU.npy', patches5)
    # for i in range(20):
    #     T = np.vstack((T, patches_new))


    # read hdf5 file
    # f = h5py.File('data', 'r')
    # list(f.keys())
    # dset = f['']





