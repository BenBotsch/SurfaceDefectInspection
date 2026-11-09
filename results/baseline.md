**HOG + linear SVM** on test: accuracy 0.644, macro F1 0.620

| Class | Precision | Recall | F1 | Images |
| --- | ---: | ---: | ---: | ---: |
| crazing | 0.520 | 0.433 | 0.473 | 30 |
| inclusion | 0.632 | 0.800 | 0.706 | 30 |
| patches | 0.571 | 0.400 | 0.471 | 30 |
| pitted_surface | 0.556 | 0.333 | 0.417 | 30 |
| rolled-in_scale | 0.682 | 1.000 | 0.811 | 30 |
| scratches | 0.794 | 0.900 | 0.844 | 30 |

Confusion matrix, rows true, columns predicted:

| | Cr | In | Pa | PS | RS | Sc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cr | 13 | 0 | 6 | 3 | 8 | 0 |
| In | 1 | 24 | 1 | 1 | 1 | 2 |
| Pa | 5 | 3 | 12 | 4 | 2 | 4 |
| PS | 6 | 9 | 1 | 10 | 3 | 1 |
| RS | 0 | 0 | 0 | 0 | 30 | 0 |
| Sc | 0 | 2 | 1 | 0 | 0 | 27 |
