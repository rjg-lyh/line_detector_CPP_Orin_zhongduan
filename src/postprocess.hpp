#pragma once

#include "preprocess.hpp"
#include "tools.hpp"
#include <vector>

using namespace cv;
using namespace std;

struct OutInfo{
    float ex;
    float e_angle;
    bool valid;
    OutInfo(){
        ex = NULL;
        e_angle = NULL;
        valid = false;
    }
    OutInfo(float x, float angle, bool v):ex(x), e_angle(angle), valid(v){}
};

struct FitInfo{
    Point2i point1;
    Point2i point2;
    bool valid;
    FitInfo(bool v):valid(v){}
    FitInfo(const Point2i &p1, const Point2i &p2, bool v):point1(p1), point2(p2), valid(v){}
};

float sigmoid(float x);

Point2i invertDot(const Point2i& point, int w, int h);

FitInfo computeEndDots(Mat& mask, cv::Scalar dot_color, vector<Point2i> v);

FitInfo justicAndInvert(const FitInfo& fitinfo, int w, int h);

FitInfo twoLines2one(FitInfo &L1, FitInfo &L2, int w, int h);

OutInfo* postprocess(Mat& src, float* pdata);

OutInfo* postprocess_no(Mat& src, float* pdata);