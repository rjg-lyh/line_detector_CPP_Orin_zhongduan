#include "core.hpp"

int build_or_inferOnePicture_FT32(const char* onnx_name, const string& path, int state){
    if(state == 0){
        if(!build_model_FT32(onnx_name)){
            return -1;
        }
        return 0;
    }


    TimerClock clock;
    cout<<setfill('-')<<setiosflags(ios::right);

    Size resize_scale =  Size(256, 256);
    size_t input_size = 1*3*256*256;
    size_t output_size = 1*4*256*256;

    Mat src = imread("img_33.png");

    // cuda预热 setfill('&')
    clock.update();
    warm_up_cuda(resize_scale);
    printInfo(clock.getTimeMilliSec(), 5, "cuda预热总时长", 0);
    // cout << PURPLE << left << setw(70)<< "|---------总预热时长: " + to_string(clock.getTimeMilliSec()) + "ms" << "|" << NONE << endl << endl;

    clock.update();
    float* pdst_pin = warpaffine_and_normalize_best_info(src, resize_scale);
    printInfo(clock.getTimeMilliSec(), 6, "前处理总时长", 0);
    
    clock.update();
    PairThree* params = load_model(path);
    nvinfer1::IExecutionContext* model = params->model;
    printInfo(clock.getTimeMilliSec(), 7, "模型加载总时长", 0);

    clock.update();
    float* output_data_pin = inference_info(model, pdst_pin, input_size, output_size);
    printInfo(clock.getTimeMilliSec(), 5, "推理总时长", 0);
    // save_data_CHW<float*>(output_data_pin, "output_data_host.txt", 256*256); //没打印全

    clock.update();
    OutInfo* outinfo = postprocess(src, output_data_pin);
    printInfo(clock.getTimeMilliSec(), 6, "后处理总时长", 0);

    cv::namedWindow("out", 0);
    cv::resizeWindow("out", cv::Size(1422, 800));
    cv::imshow("out", src);
	cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFreeHost(output_data_pin);
    delete outinfo;
    delete params;

    return 0;
}

int build_or_inferOnePicture_INT8(const char* onnx_name, const string& path, int state){
    if(state == 0){
        if(!build_model_INT8(onnx_name)){
            return -1;
        }
        return 0;
    }

    TimerClock clock;
    cout<<setfill('-')<<setiosflags(ios::right);

    Size resize_scale =  Size(256, 256);
    size_t input_size = 1*3*256*256;
    size_t output_size = 1*4*256*256;

    Mat src = imread("img_33.png");

    // cuda预热 setfill('&')
    clock.update();
    warm_up_cuda(resize_scale);
    printInfo(clock.getTimeMilliSec(), 5, "cuda预热总时长", 0);
    // cout << PURPLE << left << setw(70)<< "|---------总预热时长: " + to_string(clock.getTimeMilliSec()) + "ms" << "|" << NONE << endl << endl;

    clock.update();
    float* pdst_pin = warpaffine_and_normalize_best_info(src, resize_scale);
    printInfo(clock.getTimeMilliSec(), 6, "前处理总时长", 0);
    
    clock.update();
    PairThree* params = load_model(path);
    nvinfer1::IExecutionContext* model = params->model;
    printInfo(clock.getTimeMilliSec(), 7, "模型加载总时长", 0);

    clock.update();
    float* output_data_pin = inference_info(model, pdst_pin, input_size, output_size);
    printInfo(clock.getTimeMilliSec(), 5, "推理总时长", 0);
    // save_data_CHW<float*>(output_data_pin, "output_data_host.txt", 256*256); //没打印全

    clock.update();
    OutInfo* outinfo = postprocess_no(src, output_data_pin);
    printInfo(clock.getTimeMilliSec(), 6, "后处理总时长", 0);

    cv::namedWindow("out", 0);
    cv::resizeWindow("out", cv::Size(1422, 800));
    cv::imshow("out", src);
	cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFreeHost(output_data_pin);
    delete outinfo;
    delete params;

    return 0;

}

int performance_test(const string& path, const string& img_path, const string& label_path){
    Size resize_scale =  Size(256, 256);
    size_t input_size = 1*3*256*256;
    size_t output_size = 1*4*256*256;

    TimerClock clock;
    cout<<setfill('-')<<setiosflags(ios::right);
    double cost_time = 0;

    PairThree* params = load_model(path);
    nvinfer1::IExecutionContext* model = params->model;

    warm_up_cuda(resize_scale); //预热

    vector<vector<size_t>> confusion_matrix;
    confusion_matrix.resize(4);
    for(int i=0; i<4; ++i)
        confusion_matrix[i].resize(4); //TN FP FN TP

    ifstream ifs_1(img_path, ios::in);
    ifstream ifs_2(label_path, ios::in);
    vector<string> images;      //测试数据集的图片路径
    vector<string> labels;      //label的图片路径
    string buf;
	while (getline(ifs_1, buf))
        images.emplace_back(buf);
    while (getline(ifs_2, buf))
        labels.emplace_back(buf);

    for(int i=0; i<images.size(); ++i){
        cout << i << " /" << images.size() << " 推理中..." << endl;
        cv::Mat image = cv::imread(images[i]);
        cv::Mat label = cv::imread(labels[i]);
        cv::resize(label, label, cv::Size(256, 256));
        float* pdst_pin = warpaffine_and_normalize_best(image, resize_scale);

        clock.update();
        float* output_data_pin = inference(model, pdst_pin, input_size, output_size);
        cost_time += clock.getTimeMilliSec();

        metric(output_data_pin, label, confusion_matrix);
        checkRuntime(cudaFreeHost(output_data_pin));
    }
    double mean_iou_1a  = double(confusion_matrix[0][3])/(confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]);
    double mean_iou_1b  = double(confusion_matrix[0][0])/(confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][0]);
    double mean_iou_2a  = double(confusion_matrix[1][3])/(confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[1][3]);
    double mean_iou_2b  = double(confusion_matrix[1][0])/(confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[1][0]);
    double mean_iou_3a  = double(confusion_matrix[2][3])/(confusion_matrix[2][1] + confusion_matrix[2][2] + confusion_matrix[2][3]);
    double mean_iou_3b  = double(confusion_matrix[2][0])/(confusion_matrix[2][1] + confusion_matrix[2][2] + confusion_matrix[2][0]);
    double mean_iou_4a  = double(confusion_matrix[3][3])/(confusion_matrix[3][1] + confusion_matrix[3][2] + confusion_matrix[3][3]);
    double mean_iou_4b  = double(confusion_matrix[3][0])/(confusion_matrix[3][1] + confusion_matrix[3][2] + confusion_matrix[3][0]);
    
    cout << endl;
    printInfo(cost_time/images.size(), 6, "推理平均时长", 0);
    printInfo(mean_iou_1a*0.75+mean_iou_1b*0.25, 4, "总作物行IOU", 0);
    printInfo(mean_iou_2a*0.75+mean_iou_2b*0.25, 4, "左作物行IOU", 0);
    printInfo(mean_iou_3a*0.75+mean_iou_3b*0.25, 4, "右作物行IOU", 0);
    printInfo(mean_iou_4a*0.75+mean_iou_4b*0.25, 3, "导航线IOU", 0);

    ifs_1.close();
    ifs_2.close();
    delete params;
    return 0;
}

int runCamera(nvinfer1::IExecutionContext *model, SerialPort* serialPort,
              cv::Size resize_scale, size_t input_data_size, size_t output_data_size,
              Camera& cam, float v_des, float L, float B){

    VideoCapture capture;
    string pipeline = "v4l2src device=/dev/video4 ! video/x-raw,format=UYVY,width=1920,height=1080, \
    framerate=30/1! videoconvert ! appsink";

    // capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // video-sink=xvimagesink sync=false
    capture.open(pipeline, cv::CAP_GSTREAMER);

    if(capture.isOpened()){
        cout << "成功开启摄像头" << endl;
    }
    else{
        cout << "打开摄像头失败..." << endl;
        return -1;
    }
    double frame_W = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double frame_H = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    cout << "w: " << frame_W << endl;
    cout << "h: " << frame_H << endl;

    Size S = Size(frame_W, frame_H);
	int fps = capture.get(CAP_PROP_FPS);
	printf("current fps : %d \n", fps);
	VideoWriter writer("/home/nvidia/frame_save/8_4.avi", CAP_ANY, fps, S, true);

	namedWindow("camera-frame", 0);
    resizeWindow("camera-frame", 1422, 800); 

    Mat frame_naive, frame;
    size_t cnt = 0;
    float min_angle = 90;
    float max_angle = -90;
    while (capture.read(frame_naive)){
        undistort(frame_naive, frame, mat_intri, coff_dis, noArray());//去畸变
        // frame = frame_naive;
        float* pdst_pin = warpaffine_and_normalize_best(frame, resize_scale); //预处理
        float* output_data_pin = inference(model, pdst_pin, input_data_size, output_data_size); //模型预测结果
        OutInfo* outinfo = postprocess_no(frame, output_data_pin); //后处理
        // OutInfo* outinfo = postprocess(frame, output_data_pin); //后处理
        imshow("camera-frame", frame);
        char c = waitKey(1);
        cout << ++cnt << endl;
        if(! outinfo->valid){
            // ++cnt;
            // if(cnt == 6){
            //     while(true){serialPort->Write(hex_to_string("FF010055800000FEFF"));} //走到尽头，刹车！！
            // }
            cout << "无效图像 ! ! ! !" << endl;
            cudaFreeHost(output_data_pin);
            delete outinfo;
            continue;
        }
        // cnt = 0;
        
        float wheelAngle = control_unit(cam, L, B, frame_H, v_des, outinfo->ex, outinfo->e_angle);    // control, wheelAngle区间范围[-90, 90]
        // float wheelAngle = control_unit(cam, L, B, 512, v_des, 10, -30);
        cout << "角度偏差: " << outinfo->e_angle << endl;
        cout << "wheelAngle: " << wheelAngle << endl;
        min_angle = min(min_angle, wheelAngle);
        max_angle = max(max_angle, wheelAngle);



        string signal = angle2signal(serialPort, wheelAngle);      //车轮角度转化为对应的Hex信号
        cout << signal << endl;
        serialPort->Write(hex_to_string(signal));    //串口输出控制信号，下位机转动车轮
        writer.write(frame);

        cudaFreeHost(output_data_pin);
        delete outinfo;
		if (c == 'q') {
			break;
		}
    }
    cout<< "min_angle:" << min_angle << endl;
    cout<< "max_angle:" << max_angle << endl;
    cv::destroyAllWindows();
    capture.release();
    writer.release();
    return 0;
}

int runRobot(const string& path, const cv::Size &resize_scale, const size_t &input_size, const size_t &output_size,
            string& port, BaudRate rate, Camera& cam, float v_des, float L, float B){

    PairThree* params = load_model(path);                                                  //加载模型
    nvinfer1::IExecutionContext* model = params->model;

    warm_up_cuda(resize_scale);                                                            //预热

    SerialPort* serialPort = new SerialPort(port, rate);
    serialPort->SetTimeout(0);                                                             //无阻塞， -1：阻塞接受
    serialPort->Open();                                                                    //打开串口
    
    serialPort->Write(hex_to_string("FF010172800000FEFF"));                    // 启动农机，速度30，正方向 

    runCamera(model, serialPort, resize_scale, input_size, output_size, cam, v_des, L, B); //运行相机，开始无人导航
                                         
    // serialPort->Close();
    while(true){serialPort->Write(hex_to_string("FF010072800000FEFF"));}  // 刹车复位，车轮摆正     

    delete serialPort;
    delete params;
    return 0;
}