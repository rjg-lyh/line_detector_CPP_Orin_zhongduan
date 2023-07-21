#include "metric.hpp"

void metric(float* pdata, const cv::Mat &label, vector<vector<size_t>> &confusion_matrix){
    // lbl1 = torch.where(lbls== 1, true_mask, false_mask).unsqueeze(1)#[2 1 256 256]  #left_main
    // lbl2 = torch.where(lbls== 2, true_mask, false_mask).unsqueeze(1)                #right_main
    // lbl3_1 = torch.where(lbls== 3, true_mask, false_mask).unsqueeze(1)              #all_rows
    // lbl4 = torch.where(lbls== 4, true_mask, false_mask).unsqueeze(1)                #navigation_line
    
    for(int i=0; i<256*256; ++i){
        int dx = i%256;
        int dy = i/256;
        int area = 256*256;
        float* p_c1 = pdata;
        float* p_c2 = p_c1 + area;
        float* p_c3 = p_c2 + area;
        float* p_c4 = p_c3 + area;
        uchar val1 = label.at<Vec3b>(dy, dx)[0];
        uchar val2 = label.at<Vec3b>(dy, dx)[1];
        uchar val3 = label.at<Vec3b>(dy, dx)[2];
        //all_rows
        int pred_1 = sigmoid(*p_c1) > 0.5;
        if(val2 == 128 || val3 == 128){
            confusion_matrix[0][1*2 + pred_1] += 1;
        }
        else{
            confusion_matrix[0][0*2 + pred_1] += 1;
        }
        //left_main
        int pred_2 = sigmoid(*p_c2) > 0.5;
        if(val2 == 0 && val3 == 128){
            confusion_matrix[1][1*2 + pred_2] += 1;
        }
        else{
            confusion_matrix[1][0*2 + pred_2] += 1;
        }
        //right_main
        int pred_3 = sigmoid(*p_c3) > 0.5;
        if(val2 == 128 && val3 == 0){
            confusion_matrix[2][1*2 + pred_3] += 1;
        }
        else{
            confusion_matrix[2][0*2 + pred_3] += 1;
        }
        //navigation_line
        int pred_4 = sigmoid(*p_c4) > 0.5;
        if(val1 == 128){
            confusion_matrix[3][1*2 + pred_4] += 1;
        }
        else{
            confusion_matrix[3][0*2 + pred_4] += 1;
        }
        ++pdata;
    }
}