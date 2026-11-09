Test accuracy on corrupted copies of the 180 test images. Level 0 is the clean image.

| Corruption | Level | Strength | HOG + linear SVM | ResNet18 | ResNet18 INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian noise | 0 | sigma 0 | 0.644 | 0.983 | 0.972 |
| Gaussian noise | 1 | sigma 5 | 0.572 | 0.900 | 0.872 |
| Gaussian noise | 2 | sigma 10 | 0.433 | 0.528 | 0.506 |
| Gaussian noise | 3 | sigma 20 | 0.350 | 0.294 | 0.322 |
| Gaussian noise | 4 | sigma 30 | 0.217 | 0.233 | 0.278 |
| Gaussian blur | 0 | sigma (px) 0 | 0.644 | 0.983 | 0.972 |
| Gaussian blur | 1 | sigma (px) 0.5 | 0.622 | 0.817 | 0.806 |
| Gaussian blur | 2 | sigma (px) 1 | 0.517 | 0.628 | 0.622 |
| Gaussian blur | 3 | sigma (px) 2 | 0.278 | 0.561 | 0.583 |
| Gaussian blur | 4 | sigma (px) 3 | 0.183 | 0.539 | 0.539 |
| Brightness | 0 | offset 0 | 0.644 | 0.983 | 0.972 |
| Brightness | 1 | offset 25 | 0.661 | 0.972 | 0.844 |
| Brightness | 2 | offset 50 | 0.617 | 0.806 | 0.644 |
| Brightness | 3 | offset 75 | 0.544 | 0.678 | 0.439 |
| Brightness | 4 | offset 100 | 0.483 | 0.561 | 0.250 |
| Contrast | 0 | factor 1 | 0.644 | 0.983 | 0.972 |
| Contrast | 1 | factor 0.75 | 0.633 | 0.928 | 0.878 |
| Contrast | 2 | factor 0.5 | 0.617 | 0.739 | 0.722 |
| Contrast | 3 | factor 0.35 | 0.539 | 0.544 | 0.556 |
| Contrast | 4 | factor 0.2 | 0.400 | 0.344 | 0.383 |
| JPEG | 0 | quality 100 | 0.644 | 0.983 | 0.972 |
| JPEG | 1 | quality 50 | 0.550 | 0.789 | 0.739 |
| JPEG | 2 | quality 25 | 0.506 | 0.650 | 0.644 |
| JPEG | 3 | quality 10 | 0.239 | 0.500 | 0.483 |
| JPEG | 4 | quality 5 | 0.167 | 0.350 | 0.361 |
