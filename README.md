# Scanner Invariant Multiple Sclerosis Lesion Segmentation from MRI

This repository contains the codes corresponding to the ISBI 2020 paper titled "[Scanner Invariant Multiple Sclerosis Lesion Segmentation from MRI](https://www2.cs.sfu.ca/~hamarneh/ecopy/isbi2020.pdf)".

If you use the code, please cite the paper:

**Scanner Invariant Multiple Sclerosis Lesion Segmentation from MRI**

*Aslani S, Murino V, Dayan M, Tam R, Sona D, Hamarneh G.* Scanner invariant multiple sclerosis lesion segmentation from MRI. In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI) 2020 Apr 3 (pp. 781-785). IEEE.

DOI: http://dx.doi.org/10.1109/ISBI45749.2020.9098721

The corresponding bibtex entry is:

```
@inproceedings{aslani2020scanner,
  title={Scanner invariant multiple sclerosis lesion segmentation from MRI},
  author={Aslani, Shahab and Murino, Vittorio and Dayan, Michael and Tam, Roger and Sona, Diego and Hamarneh, Ghassan},
  booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
  pages={781--785},
  year={2020},
  organization={IEEE},
  doi={10.1109/ISBI45749.2020.9098721},
  url={http://dx.doi.org/10.1109/ISBI45749.2020.9098721}
}
```
---

### Requirements 

Numpy, Keras, Tensorflow, sklearn, h5py, nibabel, os, FSL, SimpleITK 

### Data preparation:

Please refer the pre-processing steps mentioned in the paper.

After pre-processing the training data, we extract 3D patches of size 64 * 64 * 64 from each modality and save them in hdf5 format along with their ground-truth segmentation and classification (institution) label (Please see extract_patches.py).

### Training

We have different scripts for the experiments mentioned in the paper. Baseline experiment settings can be found in base_net.py and base_dropout_net.py. The proposed scanner invariance regularization methods- correlation loss, randomized entropy loss, and uniform entropy loss respectively can be found in correlation_net.py, entropy_randomized_net.py, and entropy_uniform_net.py. To run an experiment, simply execute the corresponding script.

### Testing

1. For a test input, first pass it through the same pre-processing steps and extract 3D patches as during training process.
2. Load the model weights saved during the training.
3. Get segmented outputs for each 3D patch.
4. Transfer the 3D patches back to the same location to create the original size brain image (utils.image_gen).
5. Measure the evaluation metrics as reported in the paper. 

---
(Please feel free to change the readme file, Shahab and Arafat. utils.py and evaluation.py need to be added)

