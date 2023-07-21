#pragma once

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt-release-8.0/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>
#include "tools.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <preprocess.hpp>

using namespace std;

typedef std::function<void(
    int current, int count, const std::vector<std::string>& files, 
    nvinfer1::Dims dims, float* ptensor
)> Int8Process;

struct PairThree{
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* model = nullptr;
    PairThree(){}
    PairThree(nvinfer1::IRuntime* r, 
              nvinfer1::ICudaEngine* e, 
              nvinfer1::IExecutionContext* m):runtime(r), engine(e), model(m){}
    ~PairThree(){
        model->destroy();
        engine->destroy();
        runtime->destroy();
    }
};

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
private:
    Int8Process preprocess_;
    vector<string> allimgs_;
    size_t batchCudaSize_ = 0;
    int cursor_ = 0;
    size_t bytes_ = 0;
    nvinfer1::Dims dims_;
    vector<string> files_;
    float* tensor_host_ = nullptr;
    float* tensor_device_ = nullptr;
    vector<uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
    
public:
    Int8EntropyCalibrator(const vector<string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess);

    // 这个构造函数，是允许从缓存数据中加载标定结果，这样不用重新读取图像处理
    Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess);

    virtual ~Int8EntropyCalibrator();

    // 想要按照多少的batch进行标定
    int getBatchSize() const noexcept;

    bool next();

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept;

    const vector<uint8_t>& getEntropyCalibratorData();

    const void* readCalibrationCache(size_t& length) noexcept;

    virtual void writeCalibrationCache(const void* cache, size_t length) noexcept;
};

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};

inline const char* severity_string(nvinfer1::ILogger::Severity t);

vector<unsigned char> load_file(const string& file);

bool build_model_FT32(const char* model_name);

bool build_model_INT8(const char* model_name);

PairThree* load_model(const string& path);

float* inference_info(nvinfer1::IExecutionContext* execution_context, float* input_data_pin, size_t input_data_size, size_t output_data_size);

float* inference(nvinfer1::IExecutionContext* execution_context, float* input_data_pin, size_t input_data_size, size_t output_data_size);