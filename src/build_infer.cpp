#include "build_infer.hpp"

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}


void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if(severity <= Severity::kINFO){
        if(severity == Severity::kWARNING){
            printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else if(severity <= Severity::kERROR){
            printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else{
            printf("%s: %s\n", severity_string(severity), msg);
        }
    }
}

Int8EntropyCalibrator::Int8EntropyCalibrator(const vector<string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess) {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->allimgs_ = imagefiles;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = false;
        files_.resize(dims.d[0]);
}

// 这个构造函数，是允许从缓存数据中加载标定结果，这样不用重新读取图像处理
Int8EntropyCalibrator::Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess) {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->entropyCalibratorData_ = entropyCalibratorData;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = true;
        files_.resize(dims.d[0]);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator(){
        if(tensor_host_ != nullptr){
            checkRuntime(cudaFreeHost(tensor_host_));
            checkRuntime(cudaFree(tensor_device_));
            tensor_host_ = nullptr;
            tensor_device_ = nullptr;
        }
 }

    // 想要按照多少的batch进行标定
int Int8EntropyCalibrator::getBatchSize() const noexcept {
        return dims_.d[0];
}

bool Int8EntropyCalibrator::next() {
        int batch_size = dims_.d[0];
        if (cursor_ + batch_size > allimgs_.size())
            return false;

        for(int i = 0; i < batch_size; ++i)
            files_[i] = allimgs_[cursor_++];

        if(tensor_host_ == nullptr){
            size_t volumn = 1;
            for(int i = 0; i < dims_.nbDims; ++i)
                volumn *= dims_.d[i];
            
            bytes_ = volumn * sizeof(float);
            checkRuntime(cudaMallocHost(&tensor_host_, bytes_));
            checkRuntime(cudaMalloc(&tensor_device_, bytes_));
        }

        preprocess_(cursor_, allimgs_.size(), files_, dims_, tensor_host_);
        checkRuntime(cudaMemcpy(tensor_device_, tensor_host_, bytes_, cudaMemcpyHostToDevice));
        return true;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
        if (!next()) return false;
        bindings[0] = tensor_device_;
        return true;
}

const vector<uint8_t>& Int8EntropyCalibrator::getEntropyCalibratorData() {
        return entropyCalibratorData_;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
        if (fromCalibratorData_) {
            length = this->entropyCalibratorData_.size();
            return this->entropyCalibratorData_.data();
        }

        length = 0;
        return nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
        entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
}


// 上一节的代码
bool build_model_FT32(const char* model_name){
    TRTLogger logger;

    // 这是基本需要的组件
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if(!parser->parseFromFile(model_name, 1)){
        printf("Failed to parse %s\n", model_name);

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    int maxBatchSize = 1;
    double workSpaceSize = 1 << 30;
    printf("Workspace Size = %.2lf MB\n", workSpaceSize / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(workSpaceSize);

    // builder->setMaxBatchSize(maxBatchSize);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];
    
    // 配置输入的最小、最优、最大的范围
    // profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 256, 256));
    // profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 256, 256));
    // profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 256, 256));
    // config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine_ft32.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}

bool build_model_INT8(const char* model_name){
    TRTLogger logger;

    // 这是基本需要的组件
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if(!parser->parseFromFile(model_name, 1)){
        printf("Failed to parse %s\n", model_name);

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    double workSpaceSize = 1 << 30;
    printf("Workspace Size = %.2lf MB\n", workSpaceSize / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(workSpaceSize);


    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    // int input_channel = input_tensor->getDimensions().d[1];
    auto input_dims = input_tensor->getDimensions();
    

    auto preprocess = [](
        int current, int count, const std::vector<std::string>& files, 
        nvinfer1::Dims dims, float* ptensor
    ){
        printf("Preprocess %d / %d\n", count, current);
        size_t volumn = dims.d[1] * dims.d[2] *dims.d[3];
        size_t bytes_ = volumn * sizeof(float);
        for(int i = 0; i < files.size(); ++i){
            cv::Mat image = cv::imread(files[i]);
            float* pdst_pin = warpaffine_and_normalize_best(image, cv::Size(dims.d[3], dims.d[2]));
            checkRuntime(cudaMemcpy(ptensor + i*volumn, pdst_pin, bytes_, cudaMemcpyHostToHost));
            checkRuntime(cudaFreeHost(pdst_pin));
        }
    };

    ifstream ifs("calib_dataset.txt", ios::in);
    vector<string> imagefiles;
    string buf;
	while (getline(ifs, buf))
	{
        cout << buf << endl;
        imagefiles.emplace_back(buf);
	}
    ifs.close();

    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // 配置int8标定数据读取工具
    Int8EntropyCalibrator *calib = new Int8EntropyCalibrator(imagefiles, input_dims, preprocess
    );
    config->setInt8Calibrator(calib);


    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine_int8.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    delete calib;
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;

}

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

PairThree* load_model(const string& path){

    PairThree* params = new PairThree;
    TRTLogger logger;
    auto engine_data = load_file(path);
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return params;
    }
    else{
        printf("Deserialize cuda engine successfully.\n");
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();

    params->runtime = runtime;
    params->engine = engine;
    params->model = execution_context;

    return params;
}

float* inference_info(nvinfer1::IExecutionContext* execution_context, float* input_data_pin, size_t input_data_size, size_t output_data_size){

    TimerClock clock;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ------------------------------ 1. 准备好要推理的数据并搬运到GPU   ----------------------------

    // float* output_data_host = new float[output_data_size];
    float* output_data_pin = nullptr;
    float* input_data_device = nullptr;
    float* output_data_device = nullptr;
    
    size_t size_in = input_data_size*sizeof(float);
    size_t size_out = output_data_size*sizeof(float);


    clock.update();
    checkRuntime(cudaMalloc(&input_data_device, size_in));
    checkRuntime(cudaMalloc(&output_data_device, size_out));
    checkRuntime(cudaMallocHost(&output_data_pin, size_out));
    printInfo(clock.getTimeMilliSec(), 4, "infer: 分配内存", 1);

    clock.update();
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_pin, size_in, cudaMemcpyHostToDevice, stream));
    printInfo(clock.getTimeMilliSec(), 4, "infer: 搬运数据pinned to device", 1);

    // 用一个指针数组指定input和output在gpu中的指针。
    float* bindings[] = {input_data_device, output_data_device};

    // ------------------------------ 2. 推理并将结果搬运回CPU   ----------------------------
    clock.update();
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    printInfo(clock.getTimeMilliSec(), 4, "infer: 模型推理", 1);

    clock.update();
    checkRuntime(cudaMemcpyAsync(output_data_pin, output_data_device, size_out, cudaMemcpyDeviceToHost, stream));
    printInfo(clock.getTimeMilliSec(), 4, "infer: 搬运数据device to pinned", 1);
    checkRuntime(cudaStreamSynchronize(stream));
    printInfo(clock.getTimeMilliSec(), 6, "infer: 等待异步统一", 1);

    // ------------------------------ 3. 释放内存 ----------------------------
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
    checkRuntime(cudaFreeHost(input_data_pin)); //释放输入的 memory_pageable

    return output_data_pin;
}


float* inference(nvinfer1::IExecutionContext* execution_context, float* input_data_pin, size_t input_data_size, size_t output_data_size){

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ------------------------------ 1. 准备好要推理的数据并搬运到GPU   ----------------------------

    // float* output_data_host = new float[output_data_size];
    float* output_data_pin = nullptr;
    float* input_data_device = nullptr;
    float* output_data_device = nullptr;
    
    size_t size_in = input_data_size*sizeof(float);
    size_t size_out = output_data_size*sizeof(float);


    checkRuntime(cudaMalloc(&input_data_device, size_in));
    checkRuntime(cudaMalloc(&output_data_device, size_out));
    checkRuntime(cudaMallocHost(&output_data_pin, size_out));

    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_pin, size_in, cudaMemcpyHostToDevice, stream));


    // 用一个指针数组指定input和output在gpu中的指针。
    float* bindings[] = {input_data_device, output_data_device};

    // ------------------------------ 2. 推理并将结果搬运回CPU   ----------------------------
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);

    checkRuntime(cudaMemcpyAsync(output_data_pin, output_data_device, size_out, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));


    // ------------------------------ 3. 释放内存 ----------------------------
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
    checkRuntime(cudaFreeHost(input_data_pin)); //释放输入的 memory_pageable

    return output_data_pin;
}

