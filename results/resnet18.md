**ResNet18** on test: accuracy 0.983, macro F1 0.983

| Class | Precision | Recall | F1 | Images |
| --- | ---: | ---: | ---: | ---: |
| crazing | 1.000 | 1.000 | 1.000 | 30 |
| inclusion | 0.935 | 0.967 | 0.951 | 30 |
| patches | 1.000 | 1.000 | 1.000 | 30 |
| pitted_surface | 1.000 | 1.000 | 1.000 | 30 |
| rolled-in_scale | 1.000 | 1.000 | 1.000 | 30 |
| scratches | 0.966 | 0.933 | 0.949 | 30 |

Confusion matrix, rows true, columns predicted:

| | Cr | In | Pa | PS | RS | Sc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cr | 30 | 0 | 0 | 0 | 0 | 0 |
| In | 0 | 29 | 0 | 0 | 0 | 1 |
| Pa | 0 | 0 | 30 | 0 | 0 | 0 |
| PS | 0 | 0 | 0 | 30 | 0 | 0 |
| RS | 0 | 0 | 0 | 0 | 30 | 0 |
| Sc | 0 | 2 | 0 | 0 | 0 | 28 |
