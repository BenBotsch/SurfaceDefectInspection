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

| | crazing | inclusio | patches | pitted | rolled-i | scratche |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crazing | 30 | 0 | 0 | 0 | 0 | 0 |
| inclusio | 0 | 29 | 0 | 0 | 0 | 1 |
| patches | 0 | 0 | 30 | 0 | 0 | 0 |
| pitted | 0 | 0 | 0 | 30 | 0 | 0 |
| rolled-i | 0 | 0 | 0 | 0 | 30 | 0 |
| scratche | 0 | 2 | 0 | 0 | 0 | 28 |
