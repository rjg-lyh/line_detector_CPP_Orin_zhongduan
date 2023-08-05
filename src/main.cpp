#include "core.hpp"

void test1(){
    string port = "/dev/ttyTHS4";
    BaudRate rate = BaudRate::B_115200;

    SerialPort* serialPort = new SerialPort(port, rate);
    serialPort->SetTimeout(0);                                                             //无阻塞， -1：阻塞接受
    serialPort->Open();

    // serialPort->Write("FF 01 01 30 80 00 00 00 FF\n");
    // serialPort->Write(string_to_hex("FF 01 01 30 80 00 00 00 FF"));                   // 启动农机，速度30，正方向 
    // serialPort->Write(hex_to_string("FF 01 01 30 80 00 00 00 FF"));
    while(1)
        serialPort->Write(hex_to_string("FF010172750000FEFF"));

    // serialPort->Write("FF 00 00 00 80 00 00 00 FF\n");
    // serialPort->Write(string_to_hex("FF 00 00 00 80 00 00 00 FF"));                      // 刹车复位，车轮摆正

    serialPort->Close();

    delete serialPort;
}

int main(){
    // test1();
    // build_or_inferOnePicture_FT32("attn_unet_fake_fresh.onnx", "engine_ft32_fake.trtmodel", 1);
    // build_or_inferOnePicture_INT8("attn_unet_fake_fresh.onnx", "engine_int8_fake.trtmodel", 1);

    // performance_test("engine_int8_fake.trtmodel", "valid_dataset.txt", "valid_label.txt");
    // performance_test("engine_int8.trtmodel", "valid_dataset.txt", "valid_label.txt");

    
    string path = "engine_ft32_fake.trtmodel";      // 推理模型源文件
    // string path = "engine_int8_fake.trtmodel";      // 推理模型源文件
    Size resize_scale =  Size(256, 256);  // resize大小
    size_t input_size = 1*3*256*256;      // 推理模型的输入大小
    size_t output_size = 1*4*256*256;     // 推理模型的输出大小

    string port = "/dev/ttyTHS4";
    // string port = "/dev/pts/3";
    BaudRate rate = BaudRate::B_115200;

    Camera cam(0, 1.35, 1.1, deg2rad(-45));  // 相机位置dx, dy, dz, camera_degree

    float v_des = 0.2;                   // 车轮转速
    float L = 1.2;                        // 车身长度
    float B = 0.57;                        // 轮间半轴长度

    runRobot(path, resize_scale, input_size, output_size, port, rate, cam, v_des, L, B); // Firing ! !
    

    return 0;
}