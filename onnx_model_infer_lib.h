#ifndef ONNX_MODEL_INFER_LIB_H
#define ONNX_MODEL_INFER_LIB_H

#ifdef _WIN32
#define ONNX_MODEL_INFER_API __declspec(dllexport)
#else
#define ONNX_MODEL_INFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

ONNX_MODEL_INFER_API int  ONNX_MODEL_INFER_Init(const wchar_t* modelPath /* e.g. L"realwaste_cnn.onnx" */);
ONNX_MODEL_INFER_API int  ONNX_MODEL_INFER_Predict(const float* inputCHW, int len /* must be 3*256*256 */,
                       int* classIndexOut /* 0‑8 */);
ONNX_MODEL_INFER_API void ONNX_MODEL_INFER_Cleanup(void);

#ifdef __cplusplus
}
#endif

#endif 
