#include "preprocess.hpp"

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void warm_up_cuda(const Size& size){
    size_t dst_size = size.width * size.height * 3;

    uint8_t* pdst_warm = nullptr;

    checkRuntime(cudaMalloc(&pdst_warm, dst_size)); // 在GPU上开辟两块空间

    checkRuntime(cudaFree(pdst_warm));
}

Mat warpaffine_and_normalize_int(const Mat& image, const Size& size){  
    /*
       建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            思路讲解：https://v.douyin.com/NhrNnVm/
            代码讲解: https://v.douyin.com/NhMv4nr/
    */        

    Mat output(size, CV_8UC3);
    uint8_t* psrc_device = nullptr;
    uint8_t* pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkRuntime(cudaMalloc(&psrc_device, src_size)); // 在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&pdst_device, dst_size));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上
    
    warp_affine_bilinear_int(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        0
    );

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}

float* warpaffine_and_normalize_float(const Mat& image, const Size& size){  
    /*
       建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            思路讲解：https://v.douyin.com/NhrNnVm/
            代码讲解: https://v.douyin.com/NhMv4nr/
    */        
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    uint8_t* psrc_device = nullptr;
    float* pdst_host = new float[dst_size];
    float* pdst_device = nullptr;
    
    checkRuntime(cudaMalloc(&psrc_device, src_size)); // 在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&pdst_device, dst_size*4));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上
    
    warp_affine_bilinear_float(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(pdst_host, pdst_device, dst_size*4, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return pdst_host;
}


float* warpaffine_and_normalize_best_info(const Mat& image, const Size& size){
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    uint8_t* psrc_device = nullptr;
    // float* pdst_host = new float[dst_size];
    float* pdst_pin = nullptr;
    float* pdst_device = nullptr;


    TimerClock clock;

    checkRuntime(cudaMalloc(&psrc_device, src_size));
    checkRuntime(cudaMalloc(&pdst_device, dst_size*4));
    checkRuntime(cudaMallocHost(&pdst_pin, dst_size*4));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上

    clock.update();
    warp_affine_bilinear_best(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );
    printInfo(clock.getTimeMilliSec(), 5, "核函数处理", 1);
    // cout << GREEN << left << setw(70)<< "|---------核函数处理: " + to_string(clock.getTimeMilliSec()) + "ms" << NONEE << endl;

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());

    clock.update();
    checkRuntime(cudaMemcpy(pdst_pin, pdst_device, dst_size*4, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    printInfo(clock.getTimeMilliSec(), 4, "搬运数据device to pinned", 1);

    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return pdst_pin;
}

float* warpaffine_and_normalize_best(const Mat& image, const Size& size){
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    uint8_t* psrc_device = nullptr;
    // float* pdst_host = new float[dst_size];
    float* pdst_pin = nullptr;
    float* pdst_device = nullptr;

    checkRuntime(cudaMalloc(&psrc_device, src_size));
    checkRuntime(cudaMalloc(&pdst_device, dst_size*4));
    checkRuntime(cudaMallocHost(&pdst_pin, dst_size*4));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上

    warp_affine_bilinear_best(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaMemcpy(pdst_pin, pdst_device, dst_size*4, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来

    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return pdst_pin;
}
