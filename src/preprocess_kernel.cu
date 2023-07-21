#include "preprocess_kernel.cuh"

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

    Norm out;
    out.type  = NormType::MeanStd;
    out.alpha = alpha;
    out.channel_type = channel_type;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std,  std,  sizeof(out.std));
    return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    out.channel_type = channel_type;
    return out;
}

Norm Norm::None(){
    return Norm();
}

// 计算仿射变换矩阵
// 计算的矩阵是填充缩放

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){
    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel_int(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix, Norm norm
){
    
    int dx = blockDim.x * blockIdx.x + threadIdx.x; 
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height)  return;

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }
    // if(norm.channel_type == ChannelType::Invert){
    //     float t = c2;
    //     c2 = c0;  c0 = t;
    // }

    // if(norm.type == NormType::MeanStd){
    //     c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
    //     c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
    //     c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
    // }else if(norm.type == NormType::AlphaBeta){
    //     c0 = c0 * norm.alpha + norm.beta;
    //     c1 = c1 * norm.alpha + norm.beta;
    //     c2 = c2 * norm.alpha + norm.beta;
    // }

    // H W C分布
    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0;
    pdst[1] = c1;
    pdst[2] = c2;

    // // C H W分布
    // int area = dst_width * dst_height;
    // uint8_t* pdst_c0 = dst + dy * dst_width + dx;
    // uint8_t* pdst_c1 = pdst_c0 + area;
    // uint8_t* pdst_c2 = pdst_c1 + area;
    // *pdst_c0 = c0;
    // *pdst_c1 = c1;
    // *pdst_c2 = c2;
}

__global__ void warp_affine_bilinear_kernel_float(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value, AffineMatrix matrix, Norm norm
){
    
    int dx = blockDim.x * blockIdx.x + threadIdx.x; 
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height)  return;

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }else{
        int y_low = floorf(src_y); // 66
        int x_low = floorf(src_x); // 66
        int y_high = y_low + 1; // 67
        int x_high = x_low + 1; //67

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        float ly    = src_y - y_low; // 0
        float lx    = src_x - x_low; // 0
        float hy    = 1 - ly; // 1
        float hx    = 1 - lx; // 1
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; // w1=1 w2=0, w3=0, w4=0
        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3; //(x_low, y_low) 即src点

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]; //v1[0]
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]; //v1[1]
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]; //v1[2]
    }
    if(norm.channel_type == ChannelType::Invert){
        float t = c2;
        c2 = c0;  c0 = t;
    }

    if(norm.type == NormType::MeanStd){
        c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
        c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
        c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
    }else if(norm.type == NormType::AlphaBeta){
        c0 = c0 * norm.alpha + norm.beta;
        c1 = c1 * norm.alpha + norm.beta;
        c2 = c2 * norm.alpha + norm.beta;
    }

    // // H W C分布
    // float* pdst = dst + dy * dst_line_size + dx * 3;
    // pdst[0] = c0;
    // pdst[1] = c1;
    // pdst[2] = c2;

    // C H W分布
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

__global__ void resize_bilinear_and_normalize_kernel(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
		float sx, float sy, Norm norm, int edge
	){
		int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= edge) return;

		int dx      = position % dst_width;
		int dy      = position / dst_width;
		float src_x = (dx + 0.5f) * sx - 0.5f;
		float src_y = (dy + 0.5f) * sy - 0.5f;
		float c0, c1, c2;

		int y_low = floorf(src_y);
		int x_low = floorf(src_x);
		int y_high = limit(y_low + 1, 0, src_height - 1);
		int x_high = limit(x_low + 1, 0, src_width - 1);
		y_low = limit(y_low, 0, src_height - 1);
		x_low = limit(x_low, 0, src_width - 1);

		int ly    = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
		int lx    = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
		int hy    = INTER_RESIZE_COEF_SCALE - ly;
		int hx    = INTER_RESIZE_COEF_SCALE - lx;
		int w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
		float* pdst = dst + dy * dst_width + dx * 3;
		uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
		uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
		uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
		uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

		c0 = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
		c1 = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
		c2 = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);

		if(norm.channel_type == ChannelType::Invert){
			float t = c2;
			c2 = c0;  c0 = t;
		}

		if(norm.type == NormType::MeanStd){
			c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
			c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
			c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
		}else if(norm.type == NormType::AlphaBeta){
			c0 = c0 * norm.alpha + norm.beta;
			c1 = c1 * norm.alpha + norm.beta;
			c2 = c2 * norm.alpha + norm.beta;
		}

		int area = dst_width * dst_height;
		float* pdst_c0 = dst + dy * dst_width + dx;
		float* pdst_c1 = pdst_c0 + area;
		float* pdst_c2 = pdst_c1 + area;
		*pdst_c0 = c0;
		*pdst_c1 = c1;
		*pdst_c2 = c2;
	}

void warp_affine_bilinear_int(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value
){
    dim3 block_size(32, 32); // blocksize最大就是1024
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);
    AffineMatrix affine;
    affine.compute(Size_cu(src_width, src_height), Size_cu(dst_width, dst_height));

    float mean[3] = {0.5, 0.5, 0.5};
    float std[3] = {0.5, 0.5, 0.5};
    float alpha = 1/255.0;

    Norm norm_mean_std = Norm::mean_std(mean, std, alpha, ChannelType::Invert);

    warp_affine_bilinear_kernel_int<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine, norm_mean_std
    );
}

void warp_affine_bilinear_float(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value
){
    dim3 block_size(32, 32); // blocksize最大就是1024
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);
    AffineMatrix affine;
    affine.compute(Size_cu(src_width, src_height), Size_cu(dst_width, dst_height));

    float mean[3] = {0.5, 0.5, 0.5};
    float std[3] = {0.5, 0.5, 0.5};
    float alpha = 1/255.0;

    Norm norm_mean_std = Norm::mean_std(mean, std, alpha, ChannelType::Invert);
    // Norm norm_mean_std;

    warp_affine_bilinear_kernel_float<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine, norm_mean_std
    );
}

void warp_affine_bilinear_best(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value
){
    float sx = src_width/float(dst_width);
    float sy = src_height/float(dst_height);
    int edge = dst_height*dst_width;

    dim3 block_size(32); // blocksize最大就是1024
    dim3 grid_size((edge + 31) / 32);
    // AffineMatrix affine;
    // affine.compute(Size_cu(src_width, src_height), Size_cu(dst_width, dst_height));

    float mean[3] = {0.5, 0.5, 0.5};
    float std[3] = {0.5, 0.5, 0.5};
    float alpha = 1/255.0;

    Norm norm_mean_std = Norm::mean_std(mean, std, alpha, ChannelType::Invert);
    // Norm norm_mean_std;

    resize_bilinear_and_normalize_kernel<<<grid_size, block_size, 0, nullptr>>>(
		src, src_line_size, src_width, src_height, 
        dst, dst_width, dst_height, 
		sx, sy, norm_mean_std, edge
	);
}