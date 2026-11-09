| Runtime | Device | Test accuracy | Model size | Median latency | p95 latency |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch CPU | Intel64 Family 6 Model 165 Stepping 5, GenuineIntel, 6 threads | 0.983 | 44.8 MB | 23.41 ms | 26.65 ms |
| PyTorch CUDA | Quadro RTX 4000 | 0.983 | 44.8 MB | 3.55 ms | 5.38 ms |
| ONNX Runtime FP32 | Intel64 Family 6 Model 165 Stepping 5, GenuineIntel, 6 threads | 0.983 | 44.7 MB | 9.66 ms | 10.51 ms |
| ONNX Runtime INT8 | Intel64 Family 6 Model 165 Stepping 5, GenuineIntel, 6 threads | 0.972 | 11.3 MB | 5.73 ms | 6.63 ms |
| OpenCV DNN (C++) | same CPU, Docker on WSL2, 12 threads | n/a | 44.7 MB | 15.92 ms | 18.90 ms |

Batch 1, one test image decoded beforehand; preprocessing and inference timed together, 200 calls after 20 warm-up calls. The C++ classifier reads the FP32 model and matches ONNX Runtime in the parity test, so its accuracy was not measured separately.
