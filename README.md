# TensorRT_ONNX

### 0. Introduction
- Goal : Convert pytorch model to TensorRT int8 model using by ONNX for use in C++ code.
- Process : Pytorch model(python) -> ONNX -> TensorRT Model(C++) -> TensorRT PTQ INT8 Model(C++)
- Sample Model : Resnet18 

---

### 1. Development Environment
- Device 
  - Windows 10 laptop
  - CPU i7-11375H
  - GPU RTX-3060
- Dependency 
  - cuda 11.4.1
  - cudnn 8.4.1
  - tensorrt 8.4.3
  - pytorch 1.13.1+cu116
  - onnx 1.13.0
  - onnxruntime-gpu 1.14.0

---

### 2. Code Scheme
```
    TensorRT_ONNX/
    ├── calib_data/                   # 100 images for ptq
    ├── data/                         # input image
    ├── Pytorch/
    │   ├─ model/                     # onnx, pth, wts files
    │   ├─ 1_resnet18_torch.py        # base pytorch model
    │   ├─ 2_resnet18_onnx_runtime.py # make onnx & onnxruntime model
    │   ├─ 3_resnet18_onnx.py         # make onnx for TRT
    │   ├─ 4_resnet18_gen_wts.py      # make weight(.wts) for api TRT model 
    │   └─ utils.py  
    ├── TensorRT_ONNX/ 
    │   ├─ Engine/                    # engine file & calibration cach table
    │   ├─ TensorRT_ONNX/
    │   │   ├─ calibrator.cpp         # for ptq
    │   │   ├─ calibrator.hpp
    │   │   ├─ logging.hpp
    │   │   ├─ main.cpp               # main code
    │   │   ├─ utils.cpp              # custom util functions
    │   │   └─ utils.hpp
    │   └─ TensorRT_ONNX.sln
    ├── LICENSE
    └── README.md
```

---

### 3. Performance Evaluation
- Comparison of calculation average execution time of 100 iteration and FPS, GPU memory usage for one image [224,224,3]

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>ONNX-RT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP32</td><td>FP32</td><td>FP16</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td> 3.68 ms</td>
			<td> 2.52 ms </td>
			<td> 1.32 ms</td>
			<td> 0.56 ms</td>
			<td> 0.41 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td> 271.14 fps</td>
			<td> 396.47 fps</td>
			<td> 757.00 fps</td>
			<td> 1797.6 fps</td>
			<td> 2444.9 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td> 1.58 GB</td>
			<td> 1.18 GB</td>
			<td> 0.31 GB</td>
			<td> 0.27 GB</td>
			<td> 0.25 GB</td>
		</tr>
	</tbody>
</table>