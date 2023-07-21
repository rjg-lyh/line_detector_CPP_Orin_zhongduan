#include "postprocess.hpp"

FitInfo computeEndDots(Mat& mask, cv::Scalar dot_color, vector<Point2i> v){
    // 判断是否合法
    if(v.size() < 10){
        return FitInfo(false);
    }

    for (int i = 0; i < v.size(); i++)
	{
		cv::circle(mask, v[i], 1, dot_color, 1, 8, 0);
	}
 
	cv::Vec4f line_para;
	cv::fitLine(v, line_para, cv::DIST_L1, 0, 1e-2, 1e-2);
 
    // 垂直
    if(line_para[0] == 0){
        return FitInfo(Point2i(line_para[2], 1), Point2i(line_para[2], 254), true);
    }

	// 正常情况
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

    double k = line_para[1] / line_para[0];
    // 计算直线的端点(y = k(x - x0) + y0)
	cv::Point2i point1, point2, point3, point4;
	point1.x = 1;
	point1.y = k * (1 - point0.x) + point0.y;
	point2.x = 20;
	point2.y = k * (20 - point0.x) + point0.y;

    return FitInfo(point1, point2, true);
}


Point2i invertDot(const Point2i& point, int w, int h){
    // (dot21[0]*(w//256),dot21[1]*(h//256)), (dot22[0]*(w//256),dot22[1]*(h//256))
    int x = point.x*(w/256.0);
    int y = point.y*(h/256.0);
    return Point2i(x, y);
}

FitInfo justicAndInvert(const FitInfo& fitinfo, int w, int h){
    Point2i point1_inv, point2_inv;
    point1_inv = invertDot(fitinfo.point1, w, h);
    point2_inv = invertDot(fitinfo.point2, w, h);
    if(point1_inv.x == point2_inv.x){
        if(point1_inv.x < 0 || point1_inv.x > (w-1))
            return FitInfo(Point2i(point1_inv.x, 1), Point2i(point1_inv.x, h-2),false);
        return FitInfo(Point2i(point1_inv.x, 1), Point2i(point1_inv.x, h-2), true);
    }
    double k = double(point2_inv.y - point1_inv.y)/(point2_inv.x - point1_inv.x);

    vector<Point2i> ans;
    Point2i p1, p2, p3, p4;
	p1.x = 1;
	p1.y = k * (1 - point1_inv.x) + point1_inv.y;
    if(p1.y >= 0 && p1.y <= (h-1)) ans.emplace_back(p1);
	p2.x = w-1;
	p2.y = k * (w-1 - point1_inv.x) + point1_inv.y;
    if(p2.y >= 0 && p2.y <= (h-1)) ans.emplace_back(p2);
    p3.y = 1;
	p3.x = (1 - point1_inv.y)/k + point1_inv.x;
    if(p3.x >= 0 && p3.x <= (w-1)) ans.emplace_back(p3);
    p4.y = h-1;
	p4.x = (h-1 - point1_inv.y)/k + point1_inv.x;
    if(p4.x >= 0 && p4.x <= (w-1)) ans.emplace_back(p4);
    if(ans.size() < 2)
        return FitInfo(p1, p2, false);
    return FitInfo(ans[0], ans[1], true);

}

FitInfo twoLines2one(FitInfo &L1, FitInfo &L2, int w, int h){
    double k1, k2, angle1, angle2, angle;
    Point2i p11, p12, p21, p22;
    if(L1.point1.x == L1.point2.x){
        L1.point1.x -= 1;
    }
    if(L2.point1.x == L2.point2.x){
        L2.point1.x -= 1;
    }
    k1 = double(L1.point1.y - L1.point2.y)/(L1.point1.x - L1.point2.x);
    angle1 = rad2deg(atan(k1)); //左内角
    angle1 = angle1 < 0 ? -angle1 : (180 - angle1);

    k2 = double(L2.point1.y - L2.point2.y)/(L2.point1.x - L2.point2.x);
    angle2 = rad2deg(atan(k2)); //右内角
    angle2 = angle2 < 0 ? (180 + angle2) : angle2;
    
    angle = 180 - (180 - angle1 - angle2)/2 - angle2;
    angle = angle < 90 ? -angle : (180 - angle);

    // 防止L1、L2平行
    while(angle1 == angle2 && angle1 == 90){
        angle1 -= 0.1;
        k1 = tan(deg2rad(angle1));
    }

    // 防止导航线为90度
    if(angle == 90){
        angle -= 0.1;
    }
    else if(angle == -90){
        angle += 0.1;
    }
    float k = tan(deg2rad(angle));

    // 求L1和L2的交点
    int x0 = (k1*L1.point1.x - k2*L2.point1.x + L2.point1.y - L1.point1.y)/(k1 - k2); 
    int y0 = k1*(x0 - L1.point1.x) + L1.point1.y;

    vector<Point2i> ans;
    Point2i p1, p2, p3, p4;
	p1.x = 1;
	p1.y = k * (1 - x0) + y0;
    if(p1.y >= 0 && p1.y <= (h-1)) ans.emplace_back(p1);
	p2.x = w-1;
	p2.y = k * (w-1 - x0) + y0;
    if(p2.y >= 0 && p2.y <= (h-1)) ans.emplace_back(p2);
    p3.y = 1;
	p3.x = (1 - y0)/k + x0;
    if(p3.x >= 0 && p3.x <= (w-1)) ans.emplace_back(p3);
    p4.y = h-1;
	p4.x = (h-1 - y0)/k + x0;
    if(p4.x >= 0 && p4.x <= (w-1)) ans.emplace_back(p4);
    if(ans.size() < 2)
        return FitInfo(p3, p4, false);
    return FitInfo(ans[0], ans[1], true);
}

OutInfo* postprocess(Mat& src, float* pdata){
    vector<Point2i> v1;
    vector<Point2i> v2;
    vector<Point2i> v3;
    vector<Point2i> v4;
    
    //求出mask
    for(int i=0; i<256*256; ++i){
        int dx = i%256;
        int dy = i/256;
        int area = 256*256;
        float* p_c1 = pdata;
        float* p_c2 = p_c1 + area;
        float* p_c3 = p_c2 + area;
        float* p_c4 = p_c3 + area;
        if(sigmoid(*p_c1) > 0.5){
            v1.emplace_back(Point2i(dx, dy));
        }
        if(sigmoid(*p_c2) > 0.5){
            v2.emplace_back(Point2i(dx, dy));
        }
        if(sigmoid(*p_c3) > 0.5){
            v3.emplace_back(Point2i(dx, dy));
        }
        if(sigmoid(*p_c4) > 0.5){
            v4.emplace_back(Point2i(dx, dy));
        }
        ++pdata;
    }
    
    Mat mask = cv::Mat::zeros(256, 256, CV_8UC3);
	//将拟合点绘制到空白图上

    FitInfo fitinfo_2 = computeEndDots(mask, cv::Scalar(0, 255, 0), v2);
    FitInfo fitinfo_3 = computeEndDots(mask, cv::Scalar(255, 0, 0), v3);
    FitInfo fitinfo_4 = computeEndDots(mask, cv::Scalar(0, 0, 255), v4);
	
    // cv::line(mask, pair2.first, pair2.second, cv::Scalar(255, 255, 0), 2, 8, 0);
    // cv::line(mask, pair3.first, pair3.second, cv::Scalar(0, 255, 255), 2, 8, 0);
	// cv::line(mask, pair4.first, pair4.second, cv::Scalar(255, 0, 255), 2, 8, 0);

    // cv::imshow("mask 256✖256", mask);
    // cv::waitKey(0);
    
    OutInfo* outinfo = new OutInfo;
    int w = src.cols;
    int h = src.rows;

    if(fitinfo_2.valid){
        FitInfo fitinfo_2_inv = justicAndInvert(fitinfo_2, w, h);
        if(fitinfo_2_inv.valid){
            // cout << "point1: " << "(" << fitinfo_2_inv.point1.x << "," << fitinfo_2_inv.point1.y << ") ";
            // cout << "point2: " << "(" << fitinfo_2_inv.point2.x << "," << fitinfo_2_inv.point2.y << ")";
            draw_dotted_line2(src, fitinfo_2_inv.point1, fitinfo_2_inv.point2, cv::Scalar(0, 0, 255), 13, 45);  //左主作物行
            // cv::line(src, fitinfo_2_inv.point1, fitinfo_2_inv.point2, cv::Scalar(0, 0, 255), 13, 8, 0); 
        }
    }
    
    if(fitinfo_3.valid){
        FitInfo fitinfo_3_inv = justicAndInvert(fitinfo_3, w, h);
        if(fitinfo_3_inv.valid){
            draw_dotted_line2(src, fitinfo_3_inv.point1, fitinfo_3_inv.point2, cv::Scalar(0, 0, 255), 13, 45);  //右主作物行
        }
    }

    cv::arrowedLine(src, Point2i(w/2, h-2), Point2i(w/2, h*4.0/5), Scalar(0,255,0),10); // 中心箭头可视化

    // 导航线拟合失败，直接返回 无效
    if(! fitinfo_4.valid){
        return outinfo;
    }
    
    outinfo->valid = true;

    FitInfo fitinfo_4_inv = justicAndInvert(fitinfo_4, w, h);
    if(fitinfo_4_inv.valid){
        draw_dotted_line2(src, fitinfo_4_inv.point1, fitinfo_4_inv.point2, cv::Scalar(255, 0, 0), 13, 45);  //导航线
    }

    int x1 = fitinfo_4_inv.point1.x;
    int y1 = fitinfo_4_inv.point1.y;
    int x2 = fitinfo_4_inv.point2.x;
    int y2 = fitinfo_4_inv.point2.y;

    int x_center = w/2;
    int x0, x_t;
    int y_t = h*(4.f/5);
    float angle;
    double k;

    if(x1 == x2){
        x0 = x1;
        x_t = x1;
        angle = 90;
    }
    else{
        k = float(y1 - y2)/(x1 - x2);
        x0 = (h - y1)/k + x1;
        x_t = (y_t - y1)/k + x1;
        angle = -rad2deg(atan(k)); //-90 ~ 90，由于y轴是倒着的，所以加个负号
    }
    outinfo->ex = x0 - x_center;
    outinfo->e_angle = angle >= 0 ? (90 - angle):-(90 + angle);

    if(x0 > 0 && x0< w){
        cv::arrowedLine(src, Point2i(x0, h-2), Point2i(x_t, y_t), Scalar(255,0,0),10); // 导航线箭头可视化
    }

    cv::putText(src,"lateral_deviation: " + to_string(outinfo->ex),Point(40,50),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    cv::putText(src,"course_deviation: " + to_string(outinfo->e_angle),Point(40,100),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    cv::putText(src,"x0: " + to_string(x0),Point(40,150),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    cv::putText(src,"angle: " + to_string(angle),Point(40,200),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    // cv::putText(src,"k: " + to_string(k),Point(40,230),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);

    return outinfo;

}

OutInfo* postprocess_no(Mat& src, float* pdata){
    vector<Point2i> v1;
    vector<Point2i> v2;
    vector<Point2i> v3;
    vector<Point2i> v4;
    
    //求出mask
    for(int i=0; i<256*256; ++i){
        int dx = i%256;
        int dy = i/256;
        int area = 256*256;
        float* p_c1 = pdata;
        float* p_c2 = p_c1 + area;
        float* p_c3 = p_c2 + area;
        float* p_c4 = p_c3 + area;
        if(sigmoid(*p_c2) > 0.5){
            v2.emplace_back(Point2i(dx, dy));        
        }
        if(sigmoid(*p_c3) > 0.5){
            v3.emplace_back(Point2i(dx, dy));
        }
        ++pdata;
    }

    Mat mask = cv::Mat::zeros(256, 256, CV_8UC3);
	//将拟合点绘制到空白图上

    FitInfo fitinfo_2 = computeEndDots(mask, cv::Scalar(0, 255, 0), v2);
    FitInfo fitinfo_3 = computeEndDots(mask, cv::Scalar(255, 0, 0), v3);
    
    OutInfo* outinfo = new OutInfo;
    int w = src.cols;
    int h = src.rows;

    // 拟合失败，直接返回 无效
    if(!fitinfo_2.valid || !fitinfo_3.valid){
        return outinfo;
    }
    outinfo->valid = true;    

    FitInfo fitinfo_2_inv = justicAndInvert(fitinfo_2, w, h);
    if(fitinfo_2_inv.valid){
        draw_dotted_line2(src, fitinfo_2_inv.point1, fitinfo_2_inv.point2, cv::Scalar(0, 0, 255), 13, 45);  //左主作物行
    }
    
    FitInfo fitinfo_3_inv = justicAndInvert(fitinfo_3, w, h);
    if(fitinfo_3_inv.valid){
        draw_dotted_line2(src, fitinfo_3_inv.point1, fitinfo_3_inv.point2, cv::Scalar(0, 0, 255), 13, 45);  //右主作物行
    }

    cv::arrowedLine(src, Point2i(w/2, h-2), Point2i(w/2, h*4.0/5), Scalar(0,255,0),10); // 中心箭头可视化
  
    FitInfo fitinfo_4_inv = twoLines2one(fitinfo_2_inv, fitinfo_3_inv, w, h);
    if(fitinfo_4_inv.valid){
        draw_dotted_line2(src, fitinfo_4_inv.point1, fitinfo_4_inv.point2, cv::Scalar(255, 0, 0), 13, 45);  //导航线
    }

    int x1 = fitinfo_4_inv.point1.x;
    int y1 = fitinfo_4_inv.point1.y;
    int x2 = fitinfo_4_inv.point2.x;
    int y2 = fitinfo_4_inv.point2.y;

    int x_center = w/2;
    int x0, x_t;
    int y_t = h*(4.f/5);
    float angle;
    double k;

    if(x1 == x2){
        x0 = x1;
        x_t = x1;
        angle = 90;
    }
    else{
        k = float(y1 - y2)/(x1 - x2);
        x0 = (h - y1)/k + x1;
        x_t = (y_t - y1)/k + x1;
        angle = -rad2deg(atan(k)); //-90 ~ 90，由于y轴是倒着的，所以加个负号
    }
    outinfo->ex = x0 - x_center;
    outinfo->e_angle = angle >= 0 ? (90 - angle):-(90 + angle);

    if(x0 > 0 && x0< w){
        cv::arrowedLine(src, Point2i(x0, h-2), Point2i(x_t, y_t), Scalar(255,0,0),10); // 导航线箭头可视化
    }

    cv::putText(src,"lateral_deviation: " + to_string(outinfo->ex),Point(40,50),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    cv::putText(src,"course_deviation: " + to_string(outinfo->e_angle),Point(40,100),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    cv::putText(src,"x0: " + to_string(x0),Point(40,150),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    cv::putText(src,"angle: " + to_string(angle),Point(40,200),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);
    // cv::putText(src,"k: " + to_string(k),Point(40,230),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),3,8);

    return outinfo;

}