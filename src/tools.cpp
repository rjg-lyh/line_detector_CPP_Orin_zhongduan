#include "tools.hpp"

void ppColor(){
    printf("This is a character control test!\n" );
    printf("[%2u]" CLEAR "CLEAR\n" NONEE, __LINE__);
    printf("[%2u]" BLACK "BLACK " L_BLACK "L_BLACK\n" NONEE, __LINE__);
    printf("[%2u]" RED "RED " L_RED "L_RED\n" NONEE, __LINE__);
    printf("[%2u]" GREEN "GREEN " L_GREEN "L_GREEN\n" NONEE, __LINE__);
    printf("[%2u]" BROWN "BROWN " YELLOW "YELLOW\n" NONEE, __LINE__);
    printf("[%2u]" BLUE "BLUE " L_BLUE "L_BLUE\n" NONEE, __LINE__);
    printf("[%2u]" PURPLE "PURPLE " L_PURPLE "L_PURPLE\n" NONEE, __LINE__);
    printf("[%2u]" CYAN "CYAN " L_CYAN "L_CYAN\n" NONEE, __LINE__);
    printf("[%2u]" GRAY "GRAY " WHITE "WHITE\n" NONEE, __LINE__);
    printf("[%2u]" BOLD "BOLD\n" NONEE, __LINE__);
    printf("[%2u]" UNDERLINE "UNDERLINE\n" NONEE, __LINE__);
    printf("[%2u]" BLINK "BLINK\n" NONEE, __LINE__);
    printf("[%2u]" REVERSE "REVERSE\n" NONEE, __LINE__);
    printf("[%2u]" HIDE "HIDE\n" NONEE, __LINE__);
}

void printInfo(double time, int count, const char* seq, int state){
    if(state == 0)
        cout << PURPLE << left << setw(70 + count - 4)<< "|---------" + string(seq) + ": " + to_string(time) + "ms" << "|" << NONEE << endl << endl;
    else
        cout << GREEN << left << setw(70 + count - 4)<< "|---------" + string(seq) + ": " + to_string(time) + "ms" << "|" << NONEE << endl;    
}

void TimerClock::update(){
    _start = high_resolution_clock::now();
}
// 获取秒
double TimerClock::getTimeSecond(){
    return getTimeMicroSec()*0.000001;
}
// 获取毫秒
double TimerClock::getTimeMilliSec(){
    return getTimeMicroSec()*0.001;
}
// 获取微秒
long long TimerClock::getTimeMicroSec(){
    return duration_cast<microseconds>(high_resolution_clock::now() - _start).count();
}

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

float wrapToPi(float theta){
    while(theta < -M_PI)
        theta += 2*M_PI;
    while(theta > M_PI)
        theta -= 2*M_PI;
    return theta;
}


float deg2rad(float deg) {
    return deg * M_PI / 180.0;
}

float rad2deg(float rad) {
    return rad * 180.0 /M_PI;
}

// 画由线组成的虚线
void draw_dotted_line2(Mat& img, Point2f p1, Point2f p2, Scalar color, int thickness, float n)
{
    float w = p2.x - p1.x, h = p2.y - p1.y;
    float l = sqrtf(w * w + h * h);
    // 矫正线长度，使线个数为奇数
    int m = l / n;
    m = m % 2 ? m : m + 1;
    n = l / m;

    circle(img, p1, 1, color, thickness); // 画起点
    circle(img, p2, 1, color, thickness); // 画终点
    // 画中间点
    if (p1.y == p2.y) //水平线：y = m
    {
        float x1 = min(p1.x, p2.x);
        float x2 = max(p1.x, p2.x);
        for (float x = x1, n1 = 2 * n; x < x2; x = x + n1)
            line(img, Point2f(x, p1.y), Point2f(x + n, p1.y), color, thickness);
    }
    else if (p1.x == p2.x) //垂直线, x = m
    {
        float y1 = min(p1.y, p2.y);
        float y2 = max(p1.y, p2.y);
        for (float y = y1, n1 = 2 * n; y < y2; y = y + n1)
            line(img, Point2f(p1.x, y), Point2f(p1.x, y + n), color, thickness);
    }
    else // 倾斜线，与x轴、y轴都不垂直或平行
    {
        // 直线方程的两点式：(y-y1)/(y2-y1)=(x-x1)/(x2-x1) -> y = (y2-y1)*(x-x1)/(x2-x1)+y1
        float n1 = n * abs(w) / l;
        float k = h / w;
        float x1 = min(p1.x, p2.x);
        float x2 = max(p1.x, p2.x);
        for (float x = x1, n2 = 2 * n1; x < x2; x = x + n2)
        {
            Point p3 = Point2f(x, k * (x - p1.x) + p1.y);
            Point p4 = Point2f(x + n1, k * (x + n1 - p1.x) + p1.y);
            line(img, p3, p4, color, thickness);
        }
    }
}

// 画由点组成的虚线
void draw_dotted_line1(Mat img, Point2f p1, Point2f p2, Scalar color, int thickness, float n)
{
    float w = p2.x - p1.x, h = p2.y - p1.y;
    float l = sqrtf(w * w + h * h);
    int m = l / n;
    n = l / m; // 矫正虚点间隔，使虚点数为整数

    circle(img, p1, 1, color, thickness); // 画起点
    circle(img, p2, 1, color, thickness); // 画终点
    // 画中间点
    if (p1.y == p2.y) // 水平线：y = m
    {
        float x1 = min(p1.x, p2.x);
        float x2 = max(p1.x, p2.x);
        for (float x = x1 + n; x < x2; x = x + n)
            circle(img, Point2f(x, p1.y), 1, color, thickness);
    }
    else if (p1.x == p2.x) // 垂直线, x = m
    {
        float y1 = min(p1.y, p2.y);
        float y2 = max(p1.y, p2.y);
        for (float y = y1 + n; y < y2; y = y + n)
            circle(img, Point2f(p1.x, y), 1, color, thickness);
    }
    else // 倾斜线，与x轴、y轴都不垂直或平行
    {
        // 直线方程的两点式：(y-y1)/(y2-y1)=(x-x1)/(x2-x1) -> y = (y2-y1)*(x-x1)/(x2-x1)+y1
        float m = n * abs(w) / l;
        float k = h / w;
        float x1 = min(p1.x, p2.x);
        float x2 = max(p1.x, p2.x);
        for (float x = x1 + m; x < x2; x = x + m)
            circle(img, Point2f(x, k * (x - p1.x) + p1.y), 1, color, thickness);
    }
}

std::string string_to_hex(const std::string& input) { 
    static const char* const lut = "0123456789ABCDEF";
    size_t len = input.length(); 
    std::string output; 
    output.reserve(2 * len); 
    for (size_t i = 0; i < len; ++i) { 
        const unsigned char c = input[i]; 
        output.push_back(lut[c >> 4]); 
        output.push_back(lut[c & 15]); 
    } 
    return output; 
}

std::string hex_to_string(const std::string& str){
    std::string result;
    for (size_t i = 0; i < str.length(); i += 2)
    {
        std::string byte = str.substr(i, 2);
        char chr = (char)(int)strtol(byte.c_str(), NULL, 16);
        result.push_back(chr);
    }
    return result;
}

string DecIntToHexStr(long long num)
{
	string str;
	long long Temp = num / 16;
	int left = num % 16;
	if (Temp > 0)
		str += DecIntToHexStr(Temp);
	if (left < 10)
		str += (left + '0');
	else
		str += ('A' + left - 10);
	return str;
}