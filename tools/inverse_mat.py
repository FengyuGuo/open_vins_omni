import numpy as np

T_CtoI = np.array(
  [-0.99970,0.01447,-0.01973,-0.02593,
-0.01965,0.00590,0.99979,-0.02217,
0.01459,0.99988,-0.00562,0.00154,
0.00000,0.00000,0.00000,1.00000])

print("T_CtoI = ", T_CtoI.reshape(4, 4))
T_CtoI = T_CtoI.reshape(4, 4)
T_ItoC = np.linalg.inv(T_CtoI)
print("T_ItoC = ", T_ItoC)

T_CtoI = np.array(
  [1.00000,-0.00182,-0.00145,0.04144,
 -0.00140,0.02831,-0.99960,0.00732,
 0.00186,0.99960,0.02831,0.00725,
 0.00000,0.00000,0.00000,1.0000
])

T_CtoI = T_CtoI.reshape(4, 4)
T_ItoC = np.linalg.inv(T_CtoI)
print("T_CtoI = ", T_CtoI)
print("T_ItoC = ", T_ItoC)

'''
ROS1Visualizer.cpp:403 cam0 intrinsics:
ROS1Visualizer.cpp:404 680.638,674.268,377.713,362.114
ROS1Visualizer.cpp:411 0.37850,-0.28223,-0.00023,-0.00646,-0.93741

ROS1Visualizer.cpp:403 cam1 intrinsics:
ROS1Visualizer.cpp:404 679.162,672.543,370.449,363.116
ROS1Visualizer.cpp:411 0.36062,-0.33359,-0.00034,-0.00189,-0.81717

'''

a=np.array([680.638,674.268,377.713,362.114])
a2=a*2
print(a2)

a=np.array([679.162,672.543,370.449,363.116])
a2=a*2
print(a2)