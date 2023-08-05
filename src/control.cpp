#include "control.hpp"

float visualServoingCtl(Camera& camera, vector<float> &desiredState, vector<float> &actualState, float v_des){
    // specifiy the acutal state for better readability
    // cout << "x: " << actualState[0] << endl;
    // float x = actualState[0]*1.34/1.37/1000;
    float x = actualState[0];
    float y = actualState[1];
    float theta = actualState[2];

    // some crazy parameters
    float lambda_x_1 = 1;                     // 正系数1
    float lambda_w_1= 0.1;                     // 正系数2
    Eigen::Vector2f lambdavec(lambda_x_1, lambda_w_1);

    // state if it is a row or a column controller
    int controller_type = 0;                    //横向控制

    float angle = camera.tilt_angle;            // 相机与水平面的夹角
    float delta_z = camera.deltaz;              // 相机距离地面的高度
    
    Eigen::Matrix<float, 3, 6> IntMat;
    // IntMat << (-sin(angle)-y*cos(angle))/delta_z, 0, x*(sin(angle)+y*cos(angle))/delta_z, x*y, -1-pow(x, 2),  y,
    //           0, -(sin(angle)+y*cos(angle))/delta_z, y*(sin(angle)+y*cos(angle))/delta_z, 1+pow(y, 2), -x*y, -x,
    //           cos(angle)*pow(cos(theta),2)/delta_z, cos(angle)*cos(theta)*sin(theta)/delta_z, 
    //           -(cos(angle)*cos(theta)*(y*sin(theta) + x*cos(theta)))/delta_z, 
    //           -(y*sin(theta) + x*cos(theta))*cos(theta), -(y*sin(theta) + x*cos(theta))*sin(theta), -1;

    double f = 4.11e-3;
    IntMat << f*(-sin(angle)-y/f*cos(angle))/delta_z, 0, x*(sin(angle)+y/f*cos(angle))/delta_z, x*y/f, (-f*f-pow(x, 2))/f,  y,
              0, -(sin(angle)+y*cos(angle))/delta_z, y*(sin(angle)+y*cos(angle))/delta_z, 1+pow(y, 2), -x*y, -x,
              cos(angle)*pow(cos(theta),2)/delta_z, cos(angle)*cos(theta)*sin(theta)/delta_z, 
              -(cos(angle)*cos(theta)*(y*sin(theta) + x*cos(theta)))/delta_z/f, 
              -(y*sin(theta) + x*cos(theta))*cos(theta)/f, -(y*sin(theta) + x*cos(theta))*sin(theta)/f, -1;
    
    float delta_y = camera.deltay;
    Eigen::Matrix<float, 6, 2> TransfMat;
    TransfMat << 0,-delta_y,
                -sin(angle),0,
                cos(angle),0,
                0,0,
                0,-cos(angle),
                0,-sin(angle);

    Eigen::Vector<float, 6> Trans_vel = TransfMat.col(0);                  //  Tv
    Eigen::Vector<float, 6> Trans_ang = TransfMat.col(1);                  //  Tw

    Eigen::Matrix<float, 2, 6> Jac; 
    Jac << IntMat.row(0), IntMat.row(2);
    Eigen::Vector2f Jac_vel = Jac*Trans_vel;

    Eigen::Vector2f Jac_ang = Jac*Trans_ang;
    auto m = pseudoInverse(Jac_ang);
    Eigen::Vector2f Jac_ang_pi(m(0), m(1));
    // cout << "Jac_ang_pi.shape: " << Jac_ang_pi.size() << endl;

    float trans_delta = actualState[controller_type] - desiredState[controller_type];    // e_x
    float ang_delta = actualState[2] - desiredState[2];                                  // e_theta
    Eigen::Vector2f delta(trans_delta, wrapToPi(ang_delta));

    Eigen::Vector2f temp = lambdavec.cwiseProduct(delta);
    float ang_fb = -Jac_ang_pi.transpose()*(temp + Jac_vel * v_des);

    return ang_fb;
}

float getWheelAngle(float w_robot, float v_des, float L, float B){
  double tmp = v_des/abs(w_robot);
  float R = tmp > B ? tmp : B + 0.0001; //R必须大于B
  // float R = tmp;
  float theta_1 = atan(L/(R-B)); //内轮
  float theta_2 = atan(L/(R+B)); //外轮
  return w_robot >= 0 ? (rad2deg(theta_1) + rad2deg(theta_2))/2 : -(rad2deg(theta_1) + rad2deg(theta_2))/2;
}

float control_unit(Camera& cam, float L, float B, float frame_height, float v_des, float e_x, float e_angle){

  // cam = Camera(1.2, 0, 1, deg2rad(-80));
  // frame_height = 512;                // 像素画面高度

  // v_des = 0.2;                      // 设定好的线速度  m/s
  // e_x= 10;                          // 横向偏差
  // e_angle = -30;            // 角偏差

  float dx = 3e-6;
  e_x = e_x*dx; // e_x转换到图像坐标系中, 100像素距离 == 0.3mm图像坐标系上的距离

  vector<float> desiredState{0.0, frame_height/2*dx, 0.0};
  vector<float> actualState{e_x, frame_height/2*dx, deg2rad(e_angle)};

  float w = visualServoingCtl(cam, desiredState, actualState, v_des);
  cout << "机器人的反馈角速度: " << w << "rad/s" << endl;

  float wheelAngle = getWheelAngle(w, v_des, L, B);
  // cout << "车轮转角：" << wheelAngle << endl;

  return wheelAngle;
}

string angle2signal(SerialPort *serialPort, float wheelAngle){
  // float prop = 39 / 90.f;
  // int m = int(-wheelAngle*prop) + 39;
  // int n = m*(144.0 - 112.0)/78 + 112;
  // string s = DecIntToHexStr(n);
  // if(s.length() < 2){
  //   s = "0" + s;
  // }
  // return "FF010125" + s + "00000000FF";
  // wheelAngle = -wheelAngle;
  // if(wheelAngle < -30){
  //   return "FF010155D00000FEFF";
  // }
  // else if(wheelAngle < 0){
  //   return "FF010155B00000FEFF";
  // }
  // else if(wheelAngle < 30){
  //   return "FF010155500000FEFF";
  // }
  // else{
  //   return "FF010155300000FEFF";
  // }

  // 70 78 | 88 90  原本速度67

  wheelAngle = -wheelAngle;

  if(wheelAngle < 0){
    if(wheelAngle > -60)
      return "FF010172880000FEFF";
    else
      return "FF010172900000FEFF";
  }
  else{
    if(wheelAngle < 70)
      return "FF010172780000FEFF";
    else
      return "FF010172700000FEFF";
  }
}