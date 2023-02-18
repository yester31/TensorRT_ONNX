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

---

### 2. Code Scheme
```
    TensorRT_ONNX/
    ├── data/ # input image
    ├── Pytorch/
    │   ├─ resnet18.py # base pytorch model
    │   ├─ resnet18_onnx.py # convert onnx
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
			<td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP32</td><td>FP16</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td> ms</td>
			<td> ms </td>
			<td> ms</td>
			<td> ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td> fps</td>
			<td> fps</td>
			<td> fps</td>
			<td> fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td> GB</td>
			<td> GB</td>
			<td> GB</td>
			<td> GB</td>
		</tr>
	</tbody>
</table>

### 4. ...