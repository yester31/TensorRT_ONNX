# TensorRT_ONNX

### 0. Introduction
- Goal : Convert pytorch model to TensorRT int8 model using by ONNX for use in C++ code.
- Process : Pytorch model(python)->ONNX->TensorRT Model(C++)->TensorRT INT8 Model(C++)
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
  - onnxruntime-gpu 1.13.1

---

### 2. Code Scheme
```
    TensorRT_ONNX/
    ├── data/ # input image
    ├── Pytorch/
    │   ├─ 1_resnet18_torch.py # base pytorch model
    │   ├─ 2_resnet18_onnx.py  # generate onnx & onnxruntime model
    │   └─ utils.py  
    ├── TensorRT_ONNX/ 
    │   ├─ TensorRT_ONNX/
    │   │   ├─ main.cpp
    │   │   └─ calibration.cpp # for int8
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
			<td><strong>Pytorch</strong></td><td><strong>ONNX-RT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP32</td><td>FP32</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td> 3.68 ms</td>
			<td> 2.52 ms </td>
			<td> ms</td>
			<td> ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td> 271.14 fps</td>
			<td> 396.47 fps</td>
			<td> fps</td>
			<td> fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td> 1.7 GB</td>
			<td> 1.3 GB</td>
			<td> GB</td>
			<td> GB</td>
		</tr>
	</tbody>
</table>

### 4. ...