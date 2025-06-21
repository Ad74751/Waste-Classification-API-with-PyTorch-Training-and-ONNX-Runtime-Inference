#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>
#include "onnxruntime_c_api.h"
#include "onnx_model_infer_lib.h"

typedef const OrtApiBase *(*GetOrtApiBaseFn)(void);
static const OrtApi *ort = NULL;
static OrtEnv *env = NULL;
static OrtSession *sess = NULL;
static OrtMemoryInfo *memInfo = NULL;

static int fail(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    return -1;
}
static void ck(OrtStatus *st)
{
    if (!st)
        return;
    fprintf(stderr, "ORT ERROR: %s\n", ort->GetErrorMessage(st));
    ort->ReleaseStatus(st);
    exit(1);
}
static int init_api(void)
{
    static int done = 0;
    if (done)
        return 0;
    HMODULE dll = LoadLibraryA("onnxruntime\\lib\\onnxruntime.dll");
    if (!dll)
        return fail("onnxruntime.dll not found");
    GetOrtApiBaseFn getBase = (GetOrtApiBaseFn)GetProcAddress(dll, "OrtGetApiBase");
    if (!getBase)
        return fail("OrtGetApiBase not found");
    ort = getBase()->GetApi(ORT_API_VERSION);
    done = 1;
    return 0;
}

ONNX_MODEL_INFER_API int ONNX_MODEL_INFER_Init(const wchar_t *modelPath)
{
    if (sess)
        return 0;
    if (init_api())
        return -1;
    ck(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "rw", &env));
    OrtSessionOptions *opts = NULL;
    ck(ort->CreateSessionOptions(&opts));
    ck(ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
    ck(ort->CreateSession(env, modelPath, opts, &sess));
    ort->ReleaseSessionOptions(opts);
    ck(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memInfo));
    return 0;
}

ONNX_MODEL_INFER_API int ONNX_MODEL_INFER_Predict(const float *inputCHW, int len, int *classIndexOut)
{
    if (!sess || !inputCHW || len != 3 * 128 * 128 || !classIndexOut)
        return -1;
    int64_t shape[4] = {1, 3, 128, 128};
    OrtValue *in = NULL;
    ck(ort->CreateTensorWithDataAsOrtValue(
        memInfo, (void *)inputCHW, len * sizeof(float),
        shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
    const char *inNames[] = {"input"};
    const char *outNames[] = {"output"};
    OrtValue *out = NULL;
    ck(ort->Run(sess, NULL, inNames, (const OrtValue *const *)&in, 1,
                outNames, 1, &out));
    float *outData = NULL;
    ck(ort->GetTensorMutableData(out, (void **)&outData));
    int best = 0;
    for (int i = 1; i < 9; ++i)
        if (outData[i] > outData[best])
            best = i;
    *classIndexOut = best;
    ort->ReleaseValue(in);
    ort->ReleaseValue(out);
    return 0;
}

ONNX_MODEL_INFER_API void ONNX_MODEL_INFER_Cleanup(void)
{
    if (sess)
    {
        ort->ReleaseSession(sess);
        sess = NULL;
    }
    if (memInfo)
    {
        ort->ReleaseMemoryInfo(memInfo);
        memInfo = NULL;
    }
    if (env)
    {
        ort->ReleaseEnv(env);
        env = NULL;
    }
}
