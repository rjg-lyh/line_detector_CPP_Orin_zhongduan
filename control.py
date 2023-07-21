import numpy as np



def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2*np.pi
    while theta > np.pi:
        theta = theta - 2*np.pi
    return theta

def visualServoingCtl (camera, desiredState, actualState, v_des):         #camera里面的参数应该都是m（由handler的157行）
    # specifiy the acutal state for better readability
    x = actualState[0]           # 按理来说是a
    y = actualState[1]           # 按理来说是H/2
    theta = actualState[2]       # 角度
    
    # some crazy parameters   
    lambda_x_1 = 10               #  自己设定的正系数
    lambda_w_1= 3000              #  自己设定的正系数
    lambdavec = np.array([lambda_x_1,lambda_w_1])
    
    # state if it is a row or a column controller
    controller_type = 0            # 用来选择 Lx还是Ly  ，   0就是选择Lx   表示横向控制
    
    # s_c = J * u = L_sc * T_r->c * u_r
    
    # Computation of the Interaction Matrix as presented by Cherubini & Chaumette
    # relates the control variables [v,w] in the camera frame to the change of the features [x,y,theta]
    angle = camera.tilt_angle         #相机与水平面的夹角
    delta_z = camera.deltaz          # 相机距离地面的高度
    IntMat = np.array([[(-np.sin(angle)-y*np.cos(angle))/delta_z, 0, x*(np.sin(angle)+y*np.cos(angle))/delta_z, x*y, -1-x**2,  y],
                       [0, -(np.sin(angle)+y*np.cos(angle))/delta_z, y*(np.sin(angle)+y*np.cos(angle))/delta_z, 1+y**2, -x*y, -x],
                       [np.cos(angle)*np.power(np.cos(theta),2)/delta_z, np.cos(angle)*np.cos(theta)*np.sin(theta)/delta_z, 
                        -(np.cos(angle)*np.cos(theta)*(y*np.sin(theta) + x*np.cos(theta)))/delta_z, 
                        -(y*np.sin(theta) + x*np.cos(theta))*np.cos(theta), -(y*np.sin(theta) + x*np.cos(theta))*np.sin(theta), -1]])

    # Computation of the transformation from the robot to the camera frame
    delta_y = camera.deltay                  # 相机与轮子的水平距离
    TransfMat = np.array([[0,-delta_y],
                         [-np.sin(angle),0],
                         [np.cos(angle),0],
                         [0,0],
                         [0,-np.cos(angle)],
                         [0,-np.sin(angle)]])
    Trans_vel = TransfMat[:,0]                  #  Tv
    Trans_ang = TransfMat[:,1]                  #  Tw

    # Computation of the Jacobi-Matrix for velocity and angular velocity and their pseudo inverse
    # The Jacobi-Matrix relates the control variables in the robot frame to the change of the features
    Jac = np.array([IntMat[controller_type,:],IntMat[2,:]])        # Ar 矩阵
    Jac_vel = np.matmul(Jac,Trans_vel)                  #  Ar    （matmul 矩阵相乘）
    # Jac_vel_pi = np.linalg.pinv([Jac_vel])
    Jac_ang = np.matmul(Jac,Trans_ang)                  #  Br
    Jac_ang_pi = np.linalg.pinv([Jac_ang])              #  求矩阵的伪逆
    
    # Compute the delta, in this case the difference between the actual state and the desired state
    trans_delta = actualState[controller_type] - desiredState[controller_type]   # e_x
    ang_delta = actualState[2] - desiredState[2]             # e_theta
    delta = np.array([trans_delta,wrapToPi(ang_delta)])
    
    # Compute the feedback control for the angular velocity
    temp = lambdavec * delta
    ang_fb = np.matmul(-Jac_ang_pi.T,(temp + Jac_vel * v_des))
    return ang_fb


class Camera:            # 相机的各个参数
    def __init__(self, id,deltax,deltay,deltaz,tilt_angle,height,width,focal_length,sensor_width):
        self.id = id
        self.deltax = deltax                   # Translation to the origin of the robot
        self.deltay = deltay
        self.deltaz = deltaz                    # Heigth of the camera
        self.tilt_angle = tilt_angle            # Tilt Angle
        self.height = height                    # Camera Intrinsics
        self.width = width
        self.focal_length = focal_length
        self.sensor_width = sensor_width


if __name__ == '__main__':

    camera = Camera(1,1.2,0,1,np.deg2rad(-80),0.96,0,0,1)    #相机参数
    imgWidth = 512                                           #图像的宽高
    P = [10]                                                 #下交点的坐标，像素坐标系
    Angle = np.deg2rad(-30)                                 #导航线夹角
    desiredState = np.array([0.0, imgWidth/2, 0.0])
    actualState = np.array([P[0], imgWidth/2, Angle])
    v_des = 0.2                                              #设定好的线速度  m/s


    w = visualServoingCtl(camera, desiredState, actualState, v_des)
    print(w)