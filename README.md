# Edge-Optimized Dermatological Diagnosis (Quantized ResNet-18)

##  Project Overview
This project addresses the challenge of deploying medical diagnostic tools in low-connectivity environments (e.g., rural clinics in West Africa). By implementing **Post-Training Static Quantization (PTQ)** on a ResNet-18 architecture, I developed a skin lesion classifier that retains diagnostic accuracy while significantly reducing computational requirements.

##  Key Engineering Outcomes
| Metric | Original (FP32) | Quantized (Int8) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Size** | 44.77 MB | 11.31 MB | **74.8% Reduction** |
| **Inference Latency** | 21.79 ms | 15.69 ms | **1.4x Speedup** |
| **Accuracy** | 72.87% | 72.82% | **-0.05% (Lossless)** |

*Benchmarks run on CPU to simulate edge device constraints.*

##  Technical Stack
* **Framework:** PyTorch (Torch.nn, Torch.quantization)
* **Architecture:** Custom ResNet-18 (modified for 28x28 input)
* **Dataset:** MedMNIST v2 (DermaMNIST) - 10,015 images, 7 classes.
* **Optimization:** Layer Fusion (Conv+BN+ReLU) & Static Quantization (fbgemm).

##  Methodology
1.  **Architecture Design:** Adapted a ResNet-18 backbone by replacing the initial 7x7 convolution with a 3x3 kernel to handle low-resolution biomedical imagery.
2.  **Training:** Trained on DermaMNIST using CrossEntropyLoss and Adam optimizer.
3.  **Optimization Pipeline:**
    * **Fusion:** Merged Convolution, BatchNorm, and ReLU layers to reduce memory access overhead.
    * **Calibration:** Used a representative dataset to determine dynamic ranges for activation quantization.
    * **Conversion:** Mapped weights from Float32 to Int8.

##  Future Scope
* Deployment to Android via **ONNX Runtime**.
* Integration with offline-first mobile application.
