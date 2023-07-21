// part1 随便写写
// Mat output(Size(256, 256), CV_8UC3);
    // // cout << output.cols << endl;
    // // cout << output.rows << endl;
    // uchar* ppout = output.data;
    // for(int i=0; i<256*256*3; ++i){
    //     *(ppout++) = *(pdst_host++);
    // }
    // // output.data = (uchar*)pdst_host;
    // imwrite("output.jpg", output);
    // Mat out(output);
    // cv::cvtColor(out, out, cv::COLOR_BGR2RGB); 
    // uchar* pout = out.data; //RGB RGB RGB...
    // float* pdst = new float[256*256*3]; //RRRRRRR GGGGGGGGG BBBBBBBB
    // for(int i=0; i<256*256; ++i){
    //     int dy  = i/256;
    //     int dx = i%256;
    //     float* pdst_c0 = pdst + dy * 256 + dx;
    //     float* pdst_c1 = pdst_c0 + 256*256;
	// 	float* pdst_c2 = pdst_c1 + 256*256;
    //     uchar* pp = pout + dy*256*3 + dx*3;
    //     float c0 = (pp[0]/255.0f - 0.5f)/0.5f;
    //     float c1 = (pp[1]/255.0f - 0.5f)/0.5f;
    //     float c2 = (pp[2]/255.0f - 0.5f)/0.5f;
    //     *pdst_c0 = c0;
	// 	*pdst_c1 = c1;
	// 	*pdst_c2 = c2;
    // }
    // save_data_CHW<float*>(pdst, "mat.txt", 256*256);
    // save_data_HWC<float*>(pdst, "pdst_host.txt", 256*256);



    // for(int i=0; i<1000; ++i){
    //     cout << pdst_host[i] << endl;
    // }
    // save_data<float*>(pdst_host, "pdst_host.txt", 1*3*256*256);
    // save_data_HWC<float*>(pdst_host, "pdst_host.txt", 256*256);
    // save_data_CHW<float*>(pdst_host, "pdst_host.txt", 256*256);
    
    // Mat out;
    // cv::resize(src, out, cv::Size(256, 256));
    // cv::cvtColor(out, out, cv::COLOR_BGR2RGB); 
    // uchar* pout = out.data; //RGB RGB RGB...
    // float* pdst = new float[256*256*3]; //RRRRRRR GGGGGGGGG BBBBBBBB
    // for(int i=0; i<256*256; ++i){
    //     int dy  = i/256;
    //     int dx = i%256;
    //     float* pdst_c0 = pdst + dy * 256 + dx;
    //     float* pdst_c1 = pdst_c0 + 256*256;
	// 	float* pdst_c2 = pdst_c1 + 256*256;
    //     uchar* pp = pout + dy*256*3 + dx*3;
    //     float c0 = (pp[0]/255.0f - 0.5f)/0.5f;
    //     float c1 = (pp[1]/255.0f - 0.5f)/0.5f;
    //     float c2 = (pp[2]/255.0f - 0.5f)/0.5f;
    //     *pdst_c0 = c0;
	// 	*pdst_c1 = c1;
	// 	*pdst_c2 = c2;
    // }
    // save_data_CHW<float*>(pdst, "mat.txt", 256*256);
    // delete []pdst;

    // cout << dst.cols << endl;
    // cout << dst.rows << endl;
    // imwrite("mat.jpg", dst);

    // save_data<uint8_t*>(dst.data, "dstMat.txt", 256*256*3);
    // ofstream ofs;
    // ofs.open("dstMat.txt", ios::out);
    // uchar* pdata = dst.data;
    // for(int i=0; i<256*256*3;i += 10){
    //     ofs << int(*(pdata++))<<"  " << int(*(pdata++))<<"  " <<int(*(pdata++)) <<endl;
    // }
    // ofs.close();
    // cout << "成功存取 " << "dstMat.txt" <<  endl;


    // if(output_data_host == nullptr){
    //     cout << "False ..." << endl;
    //     return -1;
    // }else{
    //     cout << "Inference Done Successfully." << endl;
    // }

    // drawLine(src, output_data_host);

    // delete []pdst_host;
    // delete []output_data_host;
    
    //imwrite("output.jpg", output);
    //printf("Done. save to output.jpg\n");


 // part 2 不使用cuda加速的前处理
    //  clock.update();
    // Mat image(src.size(), src.type());
    // cv::resize(src, image, cv::Size(256, 256));
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // uchar* pout = image.data; //RGB RGB RGB...
    // float* pdst_host = new float[256*256*3]; //RRRRRRR GGGGGGGGG BBBBBBBB
    // for(int i=0; i<256*256; ++i){
    //     int dy  = i/256;
    //     int dx = i%256;
    //     float* pdst_c0 = pdst_host + dy * 256 + dx;
    //     float* pdst_c1 = pdst_c0 + 256*256;
	// 	float* pdst_c2 = pdst_c1 + 256*256;
    //     uchar* pp = pout + dy*256*3 + dx*3;
    //     float c0 = (pp[0]/255.0f - 0.5f)/0.5f;
    //     float c1 = (pp[1]/255.0f - 0.5f)/0.5f;
    //     float c2 = (pp[2]/255.0f - 0.5f)/0.5f;
    //     *pdst_c0 = c0;
	// 	*pdst_c1 = c1;
	// 	*pdst_c2 = c2;
    // }
    // cout << PURPLE << "---------前处理时长: "<< clock.getTimeMilliSec() << "ms" << "--------"<< NONE << endl;