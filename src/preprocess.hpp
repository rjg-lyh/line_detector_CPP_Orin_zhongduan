#pragma once

#include "preprocess_kernel.cuh"
#include "tools.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>

using namespace cv;
using namespace std;

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

void warm_up_cuda(const Size& size);

Mat warpaffine_and_normalize_int(const Mat& image, const Size& size);

float* warpaffine_and_normalize_float(const Mat& image, const Size& size);

float* warpaffine_and_normalize_best(const Mat& image, const Size& size);

float* warpaffine_and_normalize_best_info(const Mat& image, const Size& size);

template<class T>
void save_data_HWC(T pdata, const char* name, size_t size){
    ofstream ofs;
    ofs.open(name, ios::out);
    for(int i=0; i<size; ++i){
        ofs << *(pdata++)<<"  " << *(pdata++)<<"  " <<*(pdata++) <<endl;
        //ofs << *(pdst_host) << endl;
    }
    ofs.close();
    cout << "成功存取 " << name <<  endl;
}

template<class T>
void save_data_CHW(T pdata, const char* name, size_t size){
    ofstream ofs;
    ofs.open(name, ios::out);
    for(int i=0; i<size; ++i){
        ofs << *(pdata)<<"  " << *(pdata + 256*256)<<"  " <<*(pdata + 256*256*2) <<endl;
        ++pdata;
        //ofs << *(pdst_host) << endl;
    }
    // for(int i=0; i<25700; ++i){
    //     ++pdata;
    // }
    // for(int i=0; i<10; ++i){
    //     ofs << *(pdata++) << endl;
    // }
    ofs.close();
    cout << "成功存取 " << name <<  endl;
}
