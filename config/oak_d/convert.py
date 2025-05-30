# guo@guo-lenovo-legion-h:~$ rosrun tf tf_echo oak_right_camera_optical_frame oak_imu_frame
# At time 0.000
# - Translation: [0.000, 0.001, -0.008]
# - Rotation: in Quaternion [0.707, 0.707, -0.000, -0.000]
#             in RPY (radian) [-3.142, -0.000, 1.571]
#             in RPY (degree) [-180.000, -0.000, 90.000]
# guo@guo-lenovo-legion-h:~$ $ rosrun tf techo oak_left_camera_optical_frame oak_imu_frame
# At time 0.000
# - Translation: [0.075, 0.001, -0.008]
# - Rotation: in Quaternion [0.707, 0.707, -0.000, -0.000]
#             in RPY (radian) [-3.142, -0.000, 1.571]
#             in RPY (degree) [-180.000, -0.000, 90.000]

import numpy as np
import quaternion

p_IinR = np.array([0.000, 0.001, -0.008])
q_ItoR = quaternion.quaternion(0.707, 0.707, -0.000, -0.000)

T_ItoR = np.eye(4)
T_ItoR[:3, :3] = quaternion.as_rotation_matrix(q_ItoR)
T_ItoR[:3, 3] = p_IinR
print(T_ItoR)

p_IinL = np.array([0.075, 0.001, -0.008])
q_ItoL = quaternion.quaternion(0.707, 0.707, -0.000, -0.000)

