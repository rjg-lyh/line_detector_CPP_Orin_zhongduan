#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tools.hpp"

using namespace std;
using namespace cv;

void metric(float* output_data_pin, const cv::Mat &label, vector<vector<size_t>> &confusion_matrix);