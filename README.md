# Waste Categorization & ONNX Inference Project

This project provides a complete pipeline for training a waste classification CNN in PyTorch and running fast inference using ONNX Runtime in C/C++ and Go Lang.

## Project Structure

- `cnn/` — Python code for model training, dataset, and utilities
- `onnxruntime/` — ONNX Runtime binaries and headers (Windows)
- `onnx_model_infer_lib.c/h` — C library for ONNX model inference
- `model.onnx` — Exported ONNX model
- `main.go` - High performance http webserver using GoLang and Fibre Library.


## Setup Instructions

### 1. Python Environment (Model Training)

1. **Install Python 3.8+** (recommend using [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).
2. Navigate to the `cnn/` directory:
   ```sh
   cd cnn
   ```
3. (Optional) Create a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```
4. Install dependencies:
   ```sh
   pip install torch torchvision numpy
   ```
5. Prepare your dataset in `cnn/dataset/raw/` (see folder structure for class folders).
6. Run training:
   ```sh
   python train.py
   ```
   This will save `model.pth` and export `model.onnx`.

### 2. ONNX Runtime (C Inference)

1. Download ONNX Runtime for Windows and extract to `onnxruntime/` (already included if you see the folder).
2. Build the C inference library:
   - Use a C compiler (e.g., MinGW-w64 GCC on Windows).
   - Example build command:
     ```sh
     gcc -Ionnxruntime/include -Lonnxruntime/lib -lonnxruntime onnx_model_infer_lib.c -o onnx_model_infer_lib.dll
     ```
   - Make sure `onnxruntime.dll` is in your PATH or next to your executable.
3. Use `ONNX_MODEL_INFER_Init`, `ONNX_MODEL_INFER_Predict`, and `ONNX_MODEL_INFER_Cleanup` from your C/C++ code to run inference.

### 3. GoLang HTTP API (main.go)

1. Install [Go](https://golang.org/dl/) (version 1.18+ recommended).
2. Install dependencies:
   ```sh
   go get github.com/gofiber/fiber/v2
   go get github.com/gofiber/fiber/v2/middleware/cors
   ```
3. Make sure `onnx_model_infer_win64_model.dll` and `model.onnx` are present in the project root.
4. Build and run the server:
   ```sh
   go run main.go
   # or to build an executable:
   go build -o waste-api.exe main.go
   ./waste-api.exe
   ```
5. The API will be available at `http://localhost:3000`.
   - Health check: `GET /health`
   - Prediction: `POST /predict` with form-data key `image` (JPEG/PNG)



### 4. Dataset
- The image data is source from:  https://www.kaggle.com/datasets/joebeachcapital/realwaste
