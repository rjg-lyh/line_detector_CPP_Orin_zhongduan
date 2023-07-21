#pragma once

#include<tools.hpp>
#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Core> 
#include<eigen3/Eigen/SVD>
#include<CppLinuxSerial/SerialPort.hpp>

using namespace mn::CppLinuxSerial;

struct Camera
{
  float deltax;               // Translation to the origin of the robot
  float deltay;
  float deltaz;               // Heigth of the camera
  float tilt_angle;           // Tilt Angle
  Camera(float dx, float dy, float dz, float angle):
  deltax(dx), deltay(dy), deltaz(dz), tilt_angle(angle){}
};

// 求伪逆矩阵
template<typename T>
Eigen::MatrixXf pseudoInverse(T matOrvec){

	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(matOrvec.data(), matOrvec.rows(), matOrvec.cols());
	// fprintf(stderr, "source matrix:\n");
	// std::cout << m << std::endl;

	// fprintf(stderr, "\nEigen implement pseudoinverse:\n");
	
	auto svd = m.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
 
	const auto &singularValues = svd.singularValues();
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> singularValuesInv(m.cols(), m.rows());
	singularValuesInv.setZero();
	double  pinvtoler = 1.e-6; // choose your tolerance wisely
	for (unsigned int i = 0; i < singularValues.size(); ++i) {
		if (singularValues(i) > pinvtoler)
			singularValuesInv(i, i) = 1.0f / singularValues(i);
		else
			singularValuesInv(i, i) = 0.f;
	}
 
	Eigen::MatrixXf pinvmat = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
  // std::cout << pinvmat << std::endl;

  return pinvmat;
}

float visualServoingCtl(Camera& camera, vector<float> &desiredState, vector<float> &actualState, float v_des);

float getWheelAngle(float w_robot, float v_des, float L, float B);

float control_unit(Camera& cam, float L, float B, float frame_height, float v_des, float e_x, float e_angle);

string angle2signal(SerialPort *serialPort, float wheelAngle);
