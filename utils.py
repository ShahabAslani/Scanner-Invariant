import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import h5py


def patch_gen(image, patch_size):
    blck = np.asanyarray(patch_size)
    counter = 0
    Total_slice = np.array([])
    for i in range(0, image.shape[0], blck[0] / 2):  # Coronal
        if i + blck[0] > image.shape[0]:  # out of range
            for j in range(0, image.shape[1], blck[0] / 2):
                if j + blck[0] > image.shape[1]:  # out of range
                    for k in range(0, image.shape[2], blck[0] / 2):
                        if k + blck[0] > image.shape[2]:  # when the patch is out of range
                            patch = image[i: image.shape[0], j: j + image.shape[1], k: image.shape[2]]
                            b = np.zeros((blck[0], blck[0], blck[0]))
                            b[:-(i + blck[0] - image.shape[0]), :-(j + blck[0] - image.shape[1]),
                            :-(k + blck[0] - image.shape[2])] = patch
                            patch = b[np.newaxis, ...]
                            Total_patch = np.vstack((Total_patch, patch))
                            counter = counter + 1
                            break
                        patch = image[i: i + image.shape[0], j: j + image.shape[1], k: k + blck[0]]
                        b = np.zeros((blck[0], blck[0], blck[0]))
                        b[:-(i + blck[0] - image.shape[0]), :-(j + blck[0] - image.shape[1]), :] = patch
                        patch = b[np.newaxis, ...]
                        if counter == 0:
                            Total_patch = patch
                            counter = counter + 1
                        else:
                            Total_patch = np.vstack((Total_patch, patch))
                            counter = counter + 1
                    break
                for k in range(0, image.shape[2], blck[0] / 2):
                    if k + blck[0] > image.shape[2]:  # out of range
                        patch = image[i: i + image.shape[0], j: j + blck[0], k: image.shape[2]]
                        b = np.zeros((blck[0], blck[0], blck[0]))
                        b[:-(i + blck[0] - image.shape[0]), :, :-(k + blck[0] - image.shape[2])] = patch
                        patch = b[np.newaxis, ...]
                        Total_patch = np.vstack((Total_patch, patch))
                        counter = counter + 1
                        break
                    patch = image[i: i + image.shape[0], j: j + blck[0], k: k + blck[0]]
                    b = np.zeros((blck[0], blck[0], blck[0]))
                    b[:-(i + blck[0] - image.shape[0]), :, :] = patch
                    patch = b[np.newaxis, ...]
                    if counter == 0:
                        Total_patch = patch
                        counter = counter + 1
                    else:
                        Total_patch = np.vstack((Total_patch, patch))
                        counter = counter + 1
            break

        for j in range(0, image.shape[1], blck[0] / 2):  # Axial
            if j + blck[0] > image.shape[1]:  # out of range
                for k in range(0, image.shape[2], blck[0] / 2):
                    if k + blck[0] > image.shape[2]:  # out of range
                        patch = image[i: i + blck[0], j: j + image.shape[1], k: image.shape[2]]
                        b = np.zeros((blck[0], blck[0], blck[0]))
                        b[:, :-(j + blck[0] - image.shape[1]), :-(k + blck[0] - image.shape[2])] = patch
                        patch = b[np.newaxis, ...]
                        Total_patch = np.vstack((Total_patch, patch))
                        counter = counter + 1
                        break
                    patch = image[i: i + blck[0], j: j + image.shape[1], k: k + blck[0]]
                    b = np.zeros((blck[0], blck[0], blck[0]))
                    b[:, :-(j + blck[0] - image.shape[1]), :] = patch
                    patch = b[np.newaxis, ...]
                    if counter == 0:
                        Total_patch = patch
                        counter = counter + 1
                    else:
                        Total_patch = np.vstack((Total_patch, patch))
                        counter = counter + 1
                break

            for k in range(0, image.shape[2], blck[0] / 2):  # Sagittal
                if k + blck[0] > image.shape[2]:  # out of range
                    patch = image[i: i + blck[0], j: j + blck[0], k: image.shape[2]]
                    b = np.zeros((blck[0], blck[0], blck[0]))
                    b[:, :, :-(k + blck[0] - image.shape[2])] = patch
                    patch = b[np.newaxis, ...]
                    Total_patch = np.vstack((Total_patch, patch))
                    counter = counter + 1
                    break
                patch = image[i: i + blck[0], j: j + blck[0], k: k + blck[0]]
                patch = patch[np.newaxis, ...]
                if counter == 0:
                    Total_patch = patch
                    counter = counter + 1
                else:
                    Total_patch = np.vstack((Total_patch, patch))
                    counter = counter + 1
    return Total_patch


def image_gen(patches, patch_size, image_size):
    image = np.zeros(image_size)
    blck = np.asanyarray(patch_size)
    counter = 0
    for i in range(0, image.shape[0], blck[0] / 2):  # Coronal
        if i + blck[0] > image.shape[0]:  # out of range
            for j in range(0, image.shape[1], blck[0] / 2):
                if j + blck[0] > image.shape[1]:  # out of range
                    for k in range(0, image.shape[2], blck[0] / 2):
                        if k + blck[0] > image.shape[2]:  # out of range
                            patch = patches[counter, :, :, :]
                            image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[
                                                                                    :-(i + blck[0] - image.shape[0]),
                                                                                    :-(j + blck[0] - image.shape[1]),
                                                                                    :-(k + blck[0] - image.shape[2])]
                            counter = counter + 1
                            break
                        patch = patches[counter, :, :, :]
                        image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[: -(i + blck[0] - image.shape[0]),
                                                                                :-(j + blck[0] - image.shape[1]), :]
                        counter = counter + 1
                    break
                for k in range(0, image.shape[2], blck[0] / 2):
                    if k + blck[0] > image.shape[2]:  # out of range
                        patch = patches[counter, :, :, :]
                        image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[:-(i + blck[0] - image.shape[0]),
                                                                                :, :-(k + blck[0] - image.shape[2])]
                        counter = counter + 1
                        break
                    patch = patches[counter, :, :, :]
                    image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[:-(i + blck[0] - image.shape[0]), :,
                                                                            :]
                    counter = counter + 1
            break

        for j in range(0, image.shape[1], blck[0] / 2):  # Axial
            if j + blck[0] > image.shape[1]:  # out of range
                for k in range(0, image.shape[2], blck[0] / 2):
                    if k + blck[0] > image.shape[2]:  # out of range
                        patch = patches[counter, :, :, :]
                        image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[:,
                                                                                :-(j + blck[0] - image.shape[1]),
                                                                                :-(k + blck[0] - image.shape[2])]
                        counter = counter + 1
                        break
                    patch = patches[counter, :, :, :]
                    image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[:, :-(j + blck[0] - image.shape[1]),
                                                                            :]
                    counter = counter + 1
                break

            for k in range(0, image.shape[2], blck[0] / 2):  # Sagittal
                if k + blck[0] > image.shape[2]:  # out of range
                    patch = patches[counter, :, :, :]
                    image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch[:, :,
                                                                            :-(k + blck[0] - image.shape[2])]
                    counter = counter + 1
                    break
                patch = patches[counter, :, :, :]
                image[i: i + blck[0], j: j + blck[0], k: k + blck[0]] = patch
                counter = counter + 1
    return image


def concat_patch(patch1, patch2, patch3, patch4):
    for i in range(patch1.shape[0]):
        # # channel first
        # B = np.vstack((patches[np.newaxis, 0, :, :, :], patches[np.newaxis, 0, :, :, :], patches[np.newaxis, 0, :, :, :]))
        # B = B[np.newaxis, ...]
        # if i == 0:
        #     Total_patch = B
        # else:
        #     Total_patch = np.vstack((Total_patch, B))

        # channel last
        B = np.concatenate((patch1[i, :, :, :, np.newaxis], patch2[i, :, :, :, np.newaxis],
                            patch3[i, :, :, :, np.newaxis], patch4[i, :, :, :, np.newaxis]), axis=3)
        B = B[np.newaxis, ...]
        if i == 0:
            Total_patch = B
        else:
            Total_patch = np.vstack((Total_patch, B))
    return Total_patch


def selective_patch(patches, labels):
    # select patches with lesions (at least one voxel)
    label_T = np.array([])
    patches_T = np.array([])
    counter = 0
    for i in range(patches.shape[0]):
        if labels[i, :, :, :, :].max() > 0:
            label = labels[i, :, :, :, :]
            label = label[np.newaxis, ...]
            patch = patches[i, :, :, :, :]
            patch = patch[np.newaxis, ...]
            if counter == 0:
                label_T = label
                patches_T = patch
                counter = counter + 1
            else:
                label_T = np.vstack((label_T, label))
                patches_T = np.vstack((patches_T, patch))
                counter = counter + 1
    return label_T, patches_T


def one_hot_vector(Mprage_list):
    for i in range(len(Mprage_list)):
        # correct id
        Scanner_list = np.empty(shape=(150, 1), dtype=int)
        Scanner_list[:, :] = int(Mprage_list[i][24:26])
        if i == 0:
            list = Scanner_list
        else:
            list = np.append(list, Scanner_list)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list)
    train_Y_one_hot = to_categorical(integer_encoded)
    return train_Y_one_hot


# selecting patches from different sites in each batch (in each batch >>> 128 patch_from 4 sites_32 patch from each
# site)
def generator(h5path, indices, batchSize, is_train, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")
    patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
    label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
    label_cl = np.empty([batchSize, number_class])
    patch_number = batchSize / 4  # 4 is the total number of scans that we got patches in each batch
    counter = 0
    for i in range(4):
        index = np.random.choice(indices, 1)
        patch = db['data%s' % index[0]][:]
        label_se_o = db['label%s_se' % index[0]][:]
        label_cl_o = db['label%s_cl' % index[0]][:]

        random_patch = np.random.choice(patch.shape[0], patch_number)
        patches[counter:counter + 4, ...] = patch[random_patch, ...]
        label_se[counter:counter + 4, ...] = label_se_o[random_patch, ...]
        label_cl[counter:counter + 4, ...] = label_cl_o[random_patch, ...]
        counter = counter + patch_number
    if is_train:
        for i in range(batchSize):
            label_cl[i] = np.random.permutation(label_cl[i])
    return patches, [label_se, label_cl]


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 3 sites_5 patch from each
# site) and apply shuffling
def generator2(h5path, indices, batchSize, is_train, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = batchSize / 3  # 4 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(3):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 5, ...] = patch[random_patch, ...]
            label_se[counter:counter + 5, ...] = label_se_o[random_patch, ...]
            label_cl[counter:counter + 5, ...] = label_cl_o[random_patch, ...]
            counter = counter + patch_number
        if is_train:
            for i in range(batchSize):
                label_cl[i] = np.random.permutation(label_cl[i])
        yield patches, [label_se, label_cl]


# selecting patches single sites in each batch with shuffling the ground truth
def generator3(h5path, indices, batchSize, is_train, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:

        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        index = np.random.choice(indices, 1)
        patch = db['data%s' % index[0]][:]
        label_se_o = db['label%s_se' % index[0]][:]
        label_cl_o = db['label%s_cl' % index[0]][:]

        random_patch = np.random.choice(patch.shape[0], batchSize)
        patches = patch[random_patch, ...]
        label_se = label_se_o[random_patch, ...]
        label_cl = label_cl_o[random_patch, ...]
        if is_train:
            for i in range(batchSize):
                label_cl[i] = np.random.permutation(label_cl[i])

        yield patches, [label_se, label_cl]


# selecting patches from single sites in each batch without shuffling the ground truth
def generator4(h5path, indices, batchSize, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        index = np.random.choice(indices, 1)
        patch = db['data%s' % index[0]][:]
        label_se_o = db['label%s_se' % index[0]][:]
        label_cl_o = db['label%s_cl' % index[0]][:]

        random_patch = np.random.choice(patch.shape[0], batchSize)
        patches = patch[random_patch, ...]
        label_se = label_se_o[random_patch, ...]
        label_cl = label_cl_o[random_patch, ...]

        yield patches, [label_se, label_cl]


# selecting patches from single sites in each batch (normalizing patches [0-1])
def generator5_norm(h5path, indices, batchSize, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        index = np.random.choice(indices, 1)
        patch = db['data%s' % index[0]][:]
        label_se_o = db['label%s_se' % index[0]][:]
        label_cl_o = db['label%s_cl' % index[0]][:]

        random_patch = np.random.choice(patch.shape[0], batchSize)
        patches = patch[random_patch, ...]
        label_se = label_se_o[random_patch, ...]
        label_cl = label_cl_o[random_patch, ...]

        yield patches / 255.0, [label_se, label_cl]


# selecting patches single sites in each batch (for normal segmentation)
def generator_seg(h5path, indices, batchSize, imagesize, channel):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        index = np.random.choice(indices, 1)
        patch = db['data%s' % index[0]][:]
        label_se_o = db['label%s_se' % index[0]][:]

        random_patch = np.random.choice(patch.shape[0], batchSize)
        patches = patch[random_patch, ...]
        label_se = label_se_o[random_patch, ...]

        yield patches, label_se


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from random sites_1 patch from each
# site)
def generator6(h5path, indices, batchSize, is_train, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = 1  # total number of scans that we got patches in each batch
        for i in range(batchSize):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[i, ...] = patch[random_patch, ...]
            label_se[i, ...] = label_se_o[random_patch, ...]
            label_cl[i, ...] = label_cl_o[random_patch, ...]
        if is_train:
            for i in range(batchSize):
                label_cl[i] = np.random.permutation(label_cl[i])
        yield patches, [label_se, label_cl]


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 5 sites_3 patch from each
# site)
def generator7(h5path, indices, batchSize, is_train, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = batchSize / 5  # 5 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(5):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 3, ...] = patch[random_patch, ...]
            label_se[counter:counter + 3, ...] = label_se_o[random_patch, ...]
            label_cl[counter:counter + 3, ...] = label_cl_o[random_patch, ...]
            counter = counter + patch_number
        if is_train:
            for i in range(batchSize):
                label_cl[i] = np.random.permutation(label_cl[i])
        yield patches, [label_se, label_cl]


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 3 sites_5 patch from each
# site)>>> without shuffling
def generator8(h5path, indices, batchSize, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = batchSize / 3  # 4 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(3):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 5, ...] = patch[random_patch, ...]
            label_se[counter:counter + 5, ...] = label_se_o[random_patch, ...]
            label_cl[counter:counter + 5, ...] = label_cl_o[random_patch, ...]
            counter = counter + patch_number
        yield patches, [label_se, label_cl]


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 3 sites_5 patch from each
# site)>>> without shuffling >>> for just single dice
def generator9(h5path, indices, batchSize, imagesize, channel):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        patch_number = batchSize / 3  # 4 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(3):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 5, ...] = patch[random_patch, ...]
            label_se[counter:counter + 5, ...] = label_se_o[random_patch, ...]
            counter = counter + patch_number
        yield patches, label_se


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 3 sites_5 patch from each
# site)>>> 1.3 vas 3 T magnet strength >>> giving 0.5 probability for each class
def generator10(h5path, indices, batchSize, imagesize, channel, number_class):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = batchSize / 3  # 4 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(3):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 5, ...] = patch[random_patch, ...]
            label_se[counter:counter + 5, ...] = label_se_o[random_patch, ...]
            label_cl[counter:counter + 5, ...] = 0.5
            counter = counter + patch_number
        yield patches, [label_se, label_cl]


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 3 sites_5 patch from each
# site)>>> 1.3 vs 3 T magnet strength >>> for correlation >>> adding one more class to handle the measuring >>> make
# first column 1 as ground-truth
def generator11(h5path, indices, batchSize, imagesize, channel, number_class, magnet_strenth):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = batchSize / 3  # 4 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(3):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 5, ...] = patch[random_patch, ...]
            label_se[counter:counter + 5, ...] = label_se_o[random_patch, ...]
            if magnet_strenth == 1.5:
                label_cl[counter:counter + 5, 0] = 1
            else:
                label_cl[counter:counter + 5, 2] = 1

            counter = counter + patch_number
        yield patches, [label_se, label_cl]


# selecting patches from different sites in each batch (in each batch >>> 15 patch_from 3 sites_5 patch from each
# site)>>> without shuffling
# we push the classification to have 10 classes by removing the classes in testing and validation according to the 7-folds
# during the training procedure, it removes the validation and testing classes according to the folds >>> creating 10 classes
# during the validation procedure, it removes some of training classes (at least 4 classes to create 10 classes), then we shoufle the 10 class ground truth
def generator12(h5path, indices, batchSize, imagesize, channel, number_class, fold, is_train, is_validation):
    db = h5py.File(h5path, "r")

    while True:
        patches = np.empty([batchSize, imagesize, imagesize, imagesize, channel])
        label_se = np.empty([batchSize, imagesize, imagesize, imagesize, 1])
        label_cl = np.empty([batchSize, number_class])
        patch_number = batchSize / 3  # 4 is the total number of scans that we got patches in each batch
        counter = 0
        for i in range(3):
            index = np.random.choice(indices, 1)
            patch = db['data%s' % index[0]][:]
            label_se_o = db['label%s_se' % index[0]][:]
            label_cl_o = db['label%s_cl' % index[0]][:]

            random_patch = np.random.choice(patch.shape[0], patch_number)
            patches[counter:counter + 5, ...] = patch[random_patch, ...]
            label_se[counter:counter + 5, ...] = label_se_o[random_patch, ...]
            label_cl[counter:counter + 5, ...] = label_cl_o[random_patch, ...]
            counter = counter + patch_number

        if is_train:
            if fold == 1:
                label_cl = np.delete(label_cl, 10, 1)
                label_cl = np.delete(label_cl, (11 - 1), 1)
                label_cl = np.delete(label_cl, (12 - 2), 1)
                label_cl = np.delete(label_cl, (13 - 3), 1)
            if fold == 2:
                label_cl = np.delete(label_cl, 0, 1)
                label_cl = np.delete(label_cl, (1 - 1), 1)
                label_cl = np.delete(label_cl, (8 - 2), 1)
                label_cl = np.delete(label_cl, (9 - 3), 1)
            if fold == 3:
                label_cl = np.delete(label_cl, 2, 1)
                label_cl = np.delete(label_cl, (3 - 1), 1)
                label_cl = np.delete(label_cl, (6 - 2), 1)
                label_cl = np.delete(label_cl, (7 - 3), 1)
            if fold == 4:
                label_cl = np.delete(label_cl, 2, 1)
                label_cl = np.delete(label_cl, (3 - 1), 1)
                label_cl = np.delete(label_cl, (4 - 2), 1)
                label_cl = np.delete(label_cl, (5 - 3), 1)
            if fold == 5:
                label_cl = np.delete(label_cl, 0, 1)
                label_cl = np.delete(label_cl, (1 - 1), 1)
                label_cl = np.delete(label_cl, (6 - 2), 1)
                label_cl = np.delete(label_cl, (7 - 3), 1)
            if fold == 6:
                label_cl = np.delete(label_cl, 0, 1)
                label_cl = np.delete(label_cl, (3 - 1), 1)
                label_cl = np.delete(label_cl, (8 - 2), 1)
                label_cl = np.delete(label_cl, (9 - 3), 1)
            if fold == 7:
                label_cl = np.delete(label_cl, 2, 1)
                label_cl = np.delete(label_cl, (6 - 1), 1)
                label_cl = np.delete(label_cl, (10 - 2), 1)
                label_cl = np.delete(label_cl, (11 - 3), 1)

        if is_validation:
            if fold == 1:
                label_cl = np.delete(label_cl, 0, 1)
                label_cl = np.delete(label_cl, (1 - 1), 1)
                label_cl = np.delete(label_cl, (2 - 2), 1)
                label_cl = np.delete(label_cl, (3 - 3), 1)
            if fold == 2:
                label_cl = np.delete(label_cl, 2, 1)
                label_cl = np.delete(label_cl, (3 - 1), 1)
                label_cl = np.delete(label_cl, (4 - 2), 1)
                label_cl = np.delete(label_cl, (5 - 3), 1)
            if fold == 3:
                label_cl = np.delete(label_cl, 0, 1)
                label_cl = np.delete(label_cl, (1 - 1), 1)
                label_cl = np.delete(label_cl, (4 - 2), 1)
                label_cl = np.delete(label_cl, (5 - 3), 1)
            if fold == 4:
                label_cl = np.delete(label_cl, 0, 1)
                label_cl = np.delete(label_cl, (1 - 1), 1)
                label_cl = np.delete(label_cl, (6 - 2), 1)
                label_cl = np.delete(label_cl, (7 - 3), 1)
            if fold == 5:
                label_cl = np.delete(label_cl, 2, 1)
                label_cl = np.delete(label_cl, (3 - 1), 1)
                label_cl = np.delete(label_cl, (4 - 2), 1)
                label_cl = np.delete(label_cl, (5 - 3), 1)
            if fold == 6:
                label_cl = np.delete(label_cl, 2, 1)
                label_cl = np.delete(label_cl, (4 - 1), 1)
                label_cl = np.delete(label_cl, (5 - 2), 1)
                label_cl = np.delete(label_cl, (6 - 3), 1)
            if fold == 7:
                label_cl = np.delete(label_cl, 3, 1)
                label_cl = np.delete(label_cl, (4 - 1), 1)
                label_cl = np.delete(label_cl, (5 - 2), 1)
                label_cl = np.delete(label_cl, (7 - 3), 1)
            for i in range(batchSize):
                label_cl[i] = np.random.permutation(label_cl[i])

        yield patches, [label_se, label_cl]