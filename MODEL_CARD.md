# Model card: ResNet18 for NEU-DET surface defects

## Model

A ResNet18 with ImageNet weights, fine-tuned to sort a greyscale image of a hot-rolled steel strip
into one of six defect classes. The first convolution takes one channel, with its pretrained
weights summed over the colour channels. The model takes a 224 x 224 image normalised with the
ImageNet grey mean and standard deviation and returns softmax probabilities. It is distributed as
PyTorch weights (`runs/model.pt`), as ONNX at opset 13 (`runs/model.onnx`) and as a statically
quantised INT8 ONNX model (`runs/model_int8.onnx`). The weights are not part of the repository.
`sdi train` produces them in about two minutes on an RTX 3050, and `sdi export` and
`sdi benchmark` write the ONNX and INT8 models.

## Intended use

The model is a reference for a reproducible pipeline from training to C++ inference, and for
studying how a defect classifier behaves under quantisation and image corruption. It is meant for
research and teaching on NEU-DET or images that are close to it.

It is not meant to make accept or reject decisions on a production line. It has only seen one
dataset from one source, it classifies whole crops that each contain one defect, and it cannot say
that an image shows no defect at all, because there is no such class.

## Data

[`KeenForgeAI/NEU-DET-corrected`](https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected)
at a pinned revision, a cleaned release of the NEU surface defect database (Song and Yan, 2013)
without duplicate images. It has 200 x 200 greyscale images in six classes (crazing, inclusion,
patches, pitted surface, rolled-in scale, scratches) and a seeded, class-stratified 80/10/10 split.
The test split has 30 images per class. Training uses flips and a brightness and contrast change
of up to 15 grey levels and 15 %.

## Evaluation

The epoch with the best validation macro F1 is kept, and the test split is read only for
evaluation. Numbers are for the 180 test images.

| Model | Test accuracy | Macro F1 |
| --- | ---: | ---: |
| HOG + linear SVM baseline | 0.644 | 0.620 |
| ResNet18, mean ± std over seeds 0, 1 and 2 | 0.993 ± 0.008 | 0.993 ± 0.008 |
| ResNet18 seed 0, ONNX Runtime INT8 | 0.972 | 0.972 |

One test image is worth 0.6 points of accuracy, so the differences between seeds are one to three
images. The three errors of the seed-0 model are confusions between inclusion and scratches.
Per-class metrics, confusion matrices and predictions are in [results/](results/).

## Robustness

Accuracy falls quickly once images differ from the training data. Gaussian noise with a standard
deviation of 10 grey levels brings the float model to 0.528, a Gaussian blur of 1 pixel to 0.628,
JPEG at quality 50 to 0.789 and a brightness offset of 50 grey levels to 0.806. The INT8 model
behaves like the float model except for brightness, where it falls to 0.644 at an offset of 50 and
to 0.250 at 100. [results/robustness.md](results/robustness.md) has every level.

## Limits

The test split comes from the same source and the same acquisition as the training data, so the
accuracy says nothing about another plant, camera or lighting. NEU-DET is close to saturated, and
the scores are likely higher than a production task would give. The Grad-CAM maps in the README
show that two of the three errors rest on the surface beside the defect. The INT8 model was
calibrated on unmodified training images, and it should be recalibrated on images that represent
the target conditions before it is used elsewhere.

## Licence

The code is under the MIT licence. The cleaned dataset release is under CC BY 4.0, and the
original database has no stated licence. Cite the original work when using the data:
K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled
steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.
