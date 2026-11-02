# Surface Defect Inspection

Classifies hot-rolled steel surface defects on the NEU surface defect database (NEU-DET) into six
classes, from a classical HOG + linear SVM baseline to a fine-tuned ResNet18. The model is exported
to ONNX and then runs in three places that must agree: PyTorch, ONNX Runtime, and a small C++
classifier built on OpenCV DNN, with its own preprocessing and a parity test against Python.

## Results

Test split of 180 images (30 per class), model chosen by validation macro F1:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| HOG + linear SVM | 0.644 | 0.620 |
| ResNet18, ImageNet weights, 15 epochs, seed 0 | 0.983 | 0.983 |
| same, exported to ONNX (ONNX Runtime, CPU) | 0.983 | 0.983 |
| same, seeds 1 and 2 | 0.994, 1.000 | 0.994, 1.000 |

- Per-class metrics and confusion matrices are in [results/](results/). The three errors of the
  seed-0 model are confusions between `inclusion` and `scratches`.
- The C++ classifier matches the ONNX Runtime probabilities to within 7.5e-8 on the parity fixture.
- Median ONNX Runtime latency is 16.4 ms per image at batch size 1 on a desktop Intel CPU
  (Comet Lake, see `results/resnet18_onnx.json`). Training takes about two minutes on an
  RTX 3050.

### What these numbers do and do not say

NEU-DET is close to saturated for classification: the validation F1 reaches 1.0 after two epochs,
and one test image is worth 0.6 points of accuracy. The differences between seeds are one to three
images and are not evidence that one run is better. The data is also easier than production
inspection: every image is a 200 x 200 crop that shows a single defect class. The baseline shows
why learned features matter here (crazing, patches and pitted surface are texture classes that HOG
separates badly). It does not show that the model would hold up on a different line, camera or
defect mix. That would need data from that line and a detection or segmentation model, not a
classifier.

## Run it

Needs [uv](https://docs.astral.sh/uv/). On Windows the CUDA 12.6 build of PyTorch is installed, on
Linux the CPU build.

```text
uv sync
uv run sdi download        # NEU-DET at a pinned revision, every image checked against its SHA-256
uv run sdi baseline        # HOG + linear SVM -> results/baseline.{json,md}
uv run sdi train           # ResNet18 -> runs/model.pt, runs/history.json
uv run sdi evaluate        # PyTorch and ONNX Runtime on the test split -> results/
uv run sdi export          # runs/model.onnx for the C++ classifier
uv run pytest              # unit tests, ONNX parity, a training smoke test on synthetic images
```

The C++ classifier needs CMake and OpenCV 4 with the DNN module (`libopencv-dev` on Ubuntu 24.04):

```text
uv run sdi parity-fixture /tmp/fixture
cmake -S cpp -B build -DSDI_FIXTURE=/tmp/fixture
cmake --build build -j
ctest --test-dir build --output-on-failure
build/sdi-classify runs/model.onnx image.jpg ...
```

Or in a container, exactly as in CI: `docker build -f cpp/Dockerfile .`

## Design

- **Data.** The images come from the cleaned release
  [`KeenForgeAI/NEU-DET-corrected`](https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected),
  which removes duplicate images and fixes a seeded, class-stratified 80/10/10 split. The
  revision is pinned in `src/sdi/dataset.py`, so every run sees the same files.
- **Model.** ResNet18 with ImageNet weights. The images are greyscale, so the first convolution
  takes one channel, with its pretrained weights summed over the three colour channels. That
  gives a grey image the same response the RGB version would get.
- **Augmentation.** Horizontal and vertical flips plus a mild brightness and contrast change. No
  90-degree rotations, because scratches and rolled-in scale follow the rolling direction.
- **Model selection** uses the validation split only. The test split is read once per evaluation.
- **ONNX.** Exported at opset 13 with a softmax head and a dynamic batch size. The exporter's
  `Identity` nodes are removed afterwards, because OpenCV 4.6 DNN (the Ubuntu 24.04 package)
  cannot import them. A test guards against that.
- **Parity.** `src/sdi/preprocess.py` and `cpp/src/classifier.cpp` implement the same resize and
  normalisation. `sdi parity-fixture` writes an untrained model, a synthetic image and the
  probabilities Python computes, so the C++ test needs neither the dataset nor a trained model.

## Layout

```text
src/sdi/      dataset, preprocessing, baseline, model, training, metrics, ONNX export, CLI
tests/        pytest
cpp/          C++ classifier, parity test, CMake, Dockerfile
results/      metrics of the runs above
```

## Data licence and citation

This repository does not contain the images. The original database has no stated licence; the
cleaned release adds its changes under CC BY 4.0. If you use the data, cite the original work:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for
> hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.

The code is under the [MIT licence](LICENSE).
