#ifndef PREPROCESS_KERNEL_CUH
#define PREPROCESS_KERNEL_CUH

#include <cuda_runtime.h>

typedef unsigned char uint8_t;

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
template<typename _T>
static __inline__ __device__ _T limit(_T value, _T low, _T high){
    return value < low ? low : (value > high ? high : value);
}

static __inline__ __device__ int resize_cast(int value){
    return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
}

struct Size_cu{
    int width = 0, height = 0;

    Size_cu() = default;
    Size_cu(int w, int h)
    :width(w), height(h){}
};

enum class NormType : int{
        None      = 0,
        MeanStd   = 1,
        AlphaBeta = 2
    };

enum class ChannelType : int{
    None          = 0,
    Invert        = 1
};

struct Norm{
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);

    // None
    static Norm None();
};

struct AffineMatrix{

    float i2d[6];      
    float d2i[6];     

    void invertAffineTransform(float imat[6], float omat[6]){
        omat[0] = 1/imat[0]; omat[1] = 0; omat[2] = 0;
        omat[3] = 0; omat[4] = 1/imat[4]; omat[5] = 0;
    }

    void compute(const Size_cu& from, const Size_cu& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        i2d[0] = scale_x;  i2d[1] = 0; i2d[2] = 0;

        i2d[3] = 0;  i2d[4] = scale_y; i2d[5] = 0;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y);

__global__ void warp_affine_bilinear_kernel_int(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix, Norm norm
);

__global__ void warp_affine_bilinear_kernel_float(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value, AffineMatrix matrix, Norm norm
);

__global__ void resize_bilinear_and_normalize_kernel(
		uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst, int dst_width, int dst_height, 
		float sx, float sy, Norm norm, int edge
	);

void warp_affine_bilinear_int(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value
);

void warp_affine_bilinear_float(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value
);

void warp_affine_bilinear_best(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value
);



#endif