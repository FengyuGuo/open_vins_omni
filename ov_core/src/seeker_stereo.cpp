#include <cmath>
#include <deque>
#include <sstream>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <map>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cam/CamRadtan.h"
#include "cam/CamOmni.h"
#include "cam/CamEqui.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/tic_toc.h"

const int VIRT_CAM_WIDTH = 640;
const int VIRT_CAM_HEIGHT = 480;
const float VIRT_CAM_F = 320.0f;

std::shared_ptr<tf::TransformBroadcaster> TfBr;
std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_br;
std::vector<geometry_msgs::TransformStamped> static_tf;
bool static_tf_published = false;

geometry_msgs::TransformStamped convert_msg(const tf::StampedTransform& tf_transform)
{
  geometry_msgs::TransformStamped msg;
  msg.header.stamp = tf_transform.stamp_;
  msg.header.frame_id = tf_transform.frame_id_;
  msg.child_frame_id = tf_transform.child_frame_id_;
  msg.transform.rotation.w = tf_transform.getRotation().getW();
  msg.transform.rotation.x = tf_transform.getRotation().getX();
  msg.transform.rotation.y = tf_transform.getRotation().getY();
  msg.transform.rotation.z = tf_transform.getRotation().getZ();

  msg.transform.translation.x = tf_transform.getOrigin().getX();
  msg.transform.translation.y = tf_transform.getOrigin().getY();
  msg.transform.translation.z = tf_transform.getOrigin().getZ();

  return msg;
}

struct VirtCamMap{
  std::string ros_topic;
  ros::Publisher img_pub;
  cv::Mat mapx;
  cv::Mat mapy;
  std::string cam_info_topic;
  sensor_msgs::CameraInfo cam_info;
  ros::Publisher cam_info_pub;
  bool publish_tf{false};
  tf::StampedTransform trans;
};

std::map<int, std::vector<VirtCamMap>> virt_cam_map;


void img_cbk(const int& cam_id, const sensor_msgs::ImageConstPtr& msg)
{
  PRINT_INFO("got image, cam id: %d\n", cam_id);
  TicToc tic;
  if(virt_cam_map.find(cam_id)!=virt_cam_map.end() && !virt_cam_map[cam_id].empty())
  {
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(msg, "bgr8");
    for(int i = 0; i < virt_cam_map[cam_id].size(); i++)
    {
      cv::Mat undist;
      cv::remap(img->image, undist, virt_cam_map[cam_id][i].mapx, virt_cam_map[cam_id][i].mapy, cv::INTER_LINEAR);
      cv_bridge::CvImage cv_img(msg->header, "bgr8", undist);
      sensor_msgs::ImagePtr pub_msg = cv_img.toImageMsg();
      // pub_msg->format=std::string("jpg");
      virt_cam_map[cam_id][i].img_pub.publish(*pub_msg);
      virt_cam_map[cam_id][i].cam_info.header.stamp = msg->header.stamp;
      virt_cam_map[cam_id][i].cam_info_pub.publish(virt_cam_map[cam_id][i].cam_info);
    }
  }
  else
  {
    PRINT_INFO("can not find virtual camera map!\n");
  }
  PRINT_INFO("remap cost %f ms\n", tic.toc());

  if(!static_tf_published)
  {
    for(int i = 0; i < static_tf.size(); i++)
    {
      static_tf[i].header.stamp = msg->header.stamp;
      static_tf_br->sendTransform(static_tf[i]);
    }
    static_tf_published = true;
  }
}

void create_virt_map(std::shared_ptr<ov_core::CamBase> cam_model, const Eigen::Matrix3d& R_Raw_to_Virt, cv::Mat& mapx, cv::Mat& mapy)
{
  mapx = cv::Mat(cv::Size(VIRT_CAM_WIDTH, VIRT_CAM_HEIGHT), CV_32FC1), mapy = cv::Mat(cv::Size(VIRT_CAM_WIDTH, VIRT_CAM_HEIGHT), CV_32FC1);
  double cx = VIRT_CAM_WIDTH / 2, cy = VIRT_CAM_HEIGHT / 2;
  double f = VIRT_CAM_F;
  std::shared_ptr<ov_core::CamOmni> cam_ptr = std::dynamic_pointer_cast<ov_core::CamOmni>(cam_model);
  for(int u = 0; u < VIRT_CAM_WIDTH; u++)
  {
    for(int v = 0; v < VIRT_CAM_HEIGHT; v++)
    {
      double u_n = (u - cx) / f;
      double v_n = (v - cy) / f;
      Eigen::Vector3d uv_norm(u_n, v_n, 1.0);
      uv_norm /= uv_norm.norm();
      Eigen::Vector3d uv_rot = R_Raw_to_Virt.transpose() * uv_norm;

      Eigen::Vector2f uv_dist = cam_ptr->distort_space(uv_rot.cast<float>());

      mapx.at<float>(v, u) = uv_dist.x();
      mapy.at<float>(v, u) = uv_dist.y();
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "seeker_stereo");
  std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");  // 私有命名空间

  std::string config_path = "/home/guo/openvins_ws/src/open_vins/config/seeker_debug/estimator_config.yaml";
  // 读取参数，优先级高于默认值
  nh->param<std::string>("config_path", config_path, config_path);
  std::cout << config_path << std::endl;

  TfBr = std::make_shared<tf::TransformBroadcaster>();

  // static tf2_ros::StaticTransformBroadcaster static_broadcaster;

  static_tf_br = std::make_shared<tf2_ros::StaticTransformBroadcaster>();

  // Load parameters
  auto parser = std::make_shared<ov_core::YamlParser>(config_path, false);
  parser->set_node_handler(nh);

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(-1);

  int num_cameras = 4;
  // parser->parse_config("max_cameras", num_cameras);
  std::vector<std::string> cam_topics;
  /// Map between camid and camera intrinsics (fx, fy, cx, cy, d1...d4, cam_w, cam_h)
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> camera_intrinsics;

  /// Map between camid and camera extrinsics (q_CtoI, p_CinI).
  std::map<size_t, Eigen::VectorXd> camera_extrinsics;
  for(int i = 0; i < num_cameras; i++)
  {
    virt_cam_map[i] = std::vector<VirtCamMap>();
    std::string topic_camera;
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", topic_camera);
    cam_topics.push_back(topic_camera);

    // Distortion model
    std::string dist_model = "radtan";
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_model", dist_model);

    // Distortion parameters
    std::vector<double> cam_calib1 = {1, 1, 0, 0};
    std::vector<double> cam_calib2 = {0, 0, 0, 0};
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "intrinsics", cam_calib1);
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_coeffs", cam_calib2);
    double xi = 0.0;
    Eigen::VectorXd cam_calib = Eigen::VectorXd::Zero(8);
    if(cam_calib1.size() == 4)
    {
      cam_calib << cam_calib1.at(0), cam_calib1.at(1), cam_calib1.at(2), cam_calib1.at(3), cam_calib2.at(0), cam_calib2.at(1),
        cam_calib2.at(2), cam_calib2.at(3);
    }
    else if(cam_calib1.size() == 5) // omni camera. xi, fx, fy, cx, cy
    {
      PRINT_DEBUG("size of camera calib1 is 5, omni camera found\n");
      cam_calib << cam_calib1.at(1), cam_calib1.at(2), cam_calib1.at(3), cam_calib1.at(4), cam_calib2.at(0), cam_calib2.at(1),
        cam_calib2.at(2), cam_calib2.at(3);
      xi = cam_calib1.at(0);
      PRINT_DEBUG("xi: %f\n", xi);
    }

    // FOV / resolution
    std::vector<int> matrix_wh = {1, 1};
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "resolution", matrix_wh);

    // Extrinsics
    Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "T_imu_cam", T_CtoI);

    // Load these into our state
    // different with VioManager!!!!
    Eigen::Matrix<double, 7, 1> cam_eigen;
    cam_eigen.block(0, 0, 4, 1) = ov_core::rot_2_quat(T_CtoI.block(0, 0, 3, 3));
    cam_eigen.block(4, 0, 3, 1) = T_CtoI.block(0, 3, 3, 1);

    // Create intrinsics model
    if (dist_model == "equidistant") {
      camera_intrinsics.insert({i, std::make_shared<ov_core::CamEqui>(matrix_wh.at(0), matrix_wh.at(1))});
      camera_intrinsics.at(i)->set_value(cam_calib);
    } else if(dist_model == "omni"){
      PRINT_DEBUG("OMNI camera!\n");
      camera_intrinsics.insert({i, std::make_shared<ov_core::CamOmni>(matrix_wh.at(0), matrix_wh.at(1))});
      camera_intrinsics.at(i)->set_value(cam_calib);
      camera_intrinsics.at(i)->set_xi(xi);
      PRINT_DEBUG("set xi: %f\n", camera_intrinsics.at(i)->get_xi());
    } else{
      camera_intrinsics.insert({i, std::make_shared<ov_core::CamRadtan>(matrix_wh.at(0), matrix_wh.at(1))});
      camera_intrinsics.at(i)->set_value(cam_calib);
    }
    camera_extrinsics.insert({i, cam_eigen});
  }
  PRINT_INFO("max cameras: %d\n", num_cameras);

  sensor_msgs::CameraInfo left_cam_info;

  left_cam_info.distortion_model = "plumb_bob";
  left_cam_info.width = 640;
  left_cam_info.height = 480;
  left_cam_info.R[0] = 1;
  left_cam_info.R[4] = 1;
  left_cam_info.R[8] = 1;
  left_cam_info.K[0] = 320;
  left_cam_info.K[2] = 320;
  left_cam_info.K[4] = 320;
  left_cam_info.K[5] = 240;
  left_cam_info.K[8] = 1;
  left_cam_info.P[0] = 320;
  left_cam_info.P[2] = 320;
  left_cam_info.P[5] = 320;
  left_cam_info.P[6] = 240;
  left_cam_info.P[10] = 1;
  left_cam_info.D = std::vector<double>(5, 0.0);

  sensor_msgs::CameraInfo right_cam_info;
  right_cam_info = left_cam_info;


  //init front camera remap 0 as left, 1 as right
  Eigen::Vector3d LtoR = camera_extrinsics.at(1).block(4, 0, 3, 1) - camera_extrinsics.at(0).block(4, 0, 3, 1);
  double front_baseline = LtoR.norm();
  PRINT_INFO("front virtual camera baseline: %f\n", front_baseline);
  Eigen::Vector3d x_Front_inI = LtoR / front_baseline;
  std::cout << "front x: " << x_Front_inI.transpose() << std::endl;
  Eigen::Vector3d y_Front_inI = Eigen::Vector3d::UnitY().cross(x_Front_inI);
  std::cout << "front y: " << y_Front_inI.transpose() << std::endl;
  Eigen::Vector3d z_Front_inI = x_Front_inI.cross(y_Front_inI);
  std::cout << "front z: " << z_Front_inI.transpose() << std::endl;
  Eigen::Matrix3d R_Front_to_I = Eigen::Matrix3d::Identity();
  R_Front_to_I.block<3, 1>(0, 0) = x_Front_inI;
  R_Front_to_I.block<3, 1>(0, 1) = y_Front_inI;
  R_Front_to_I.block<3, 1>(0, 2) = z_Front_inI;
  std::cout << "R front to imu: \n" << R_Front_to_I << std::endl;
  Eigen::Matrix3d R_LtoI = ov_core::quat_2_Rot(camera_extrinsics.at(0).block(0, 0, 4, 1));
  std::cout << "R left to imu: " << std::endl << R_LtoI << std::endl;

  Eigen::Matrix3d R_LtoFront = R_Front_to_I.transpose() * R_LtoI;
  //                           R_I_to_Front
  std::cout << "R left to front:\n" << R_LtoFront << std::endl;
  cv::Mat mapx_Front_in_L, mapy_Front_in_L;
  create_virt_map(camera_intrinsics.at(0), R_LtoFront, mapx_Front_in_L, mapy_Front_in_L);
  VirtCamMap front_in_L;
  front_in_L.ros_topic = std::string("virt_front/left/image_raw");
  front_in_L.mapx = mapx_Front_in_L;
  front_in_L.mapy = mapy_Front_in_L;
  front_in_L.img_pub = nh->advertise<sensor_msgs::Image>(front_in_L.ros_topic, 2);
  front_in_L.cam_info_topic = std::string("virt_front/left/camera_info");
  front_in_L.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(front_in_L.cam_info_topic, 2);
  left_cam_info.header.frame_id = "depth0";
  front_in_L.cam_info = left_cam_info;
  front_in_L.publish_tf = true;
  Eigen::Quaterniond q_FronttoI(R_Front_to_I);
  tf::Quaternion quat(q_FronttoI.coeffs()(0), q_FronttoI.coeffs()(1), q_FronttoI.coeffs()(2), q_FronttoI.coeffs()(3));
  Eigen::Vector3d p_LinI = camera_extrinsics.at(0).block(4, 0, 3, 1);
  tf::Vector3 orig(p_LinI.x(), p_LinI.y(), p_LinI.z());
  front_in_L.trans.setOrigin(orig);
  front_in_L.trans.setRotation(quat);
  front_in_L.trans.child_frame_id_ = "depth0";
  front_in_L.trans.frame_id_ = "imu";
  front_in_L.trans.stamp_ = ros::Time::now();
  static_tf.push_back(convert_msg(front_in_L.trans));
  // static_broadcaster.sendTransform(convert_msg(front_in_L.trans));
  virt_cam_map[0].push_back(front_in_L);

  Eigen::Matrix3d R_RtoI = ov_core::quat_2_Rot(camera_extrinsics.at(1).block(0, 0, 4, 1));
  std::cout << "R right to imu: " << std::endl << R_RtoI << std::endl;
  Eigen::Matrix3d R_RtoFront = R_Front_to_I.transpose() * R_RtoI;
  std::cout << "R right to front:\n" << R_RtoFront << std::endl;
  cv::Mat mapx_Front_in_R, mapy_Front_in_R;
  create_virt_map(camera_intrinsics.at(1), R_RtoFront, mapx_Front_in_R, mapy_Front_in_R);
  VirtCamMap front_in_R;
  front_in_R.ros_topic = std::string("virt_front/right/image_raw");
  front_in_R.mapx = mapx_Front_in_R;
  front_in_R.mapy = mapy_Front_in_R;
  front_in_R.img_pub = nh->advertise<sensor_msgs::Image>(front_in_R.ros_topic, 2);
  front_in_R.cam_info_topic = std::string("virt_front/right/camera_info");
  front_in_R.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(front_in_R.cam_info_topic, 2);
  right_cam_info.P[3] = -front_baseline * VIRT_CAM_F;
  right_cam_info.header.frame_id = "depth0";
  front_in_R.cam_info = right_cam_info;
  virt_cam_map[1].push_back(front_in_R);

  Eigen::Vector3d RtoBR = camera_extrinsics.at(2).block(4, 0, 3, 1) - camera_extrinsics.at(1).block(4, 0, 3, 1);
  double right_baseline = RtoBR.norm();
  PRINT_DEBUG("right virtual camera baseline: %f\n", right_baseline);
  Eigen::Vector3d x_right_inI = RtoBR / right_baseline;
  Eigen::Vector3d y_right_inI = Eigen::Vector3d::UnitX().cross(x_right_inI);
  Eigen::Vector3d z_right_inI = x_right_inI.cross(y_right_inI);

  Eigen::Matrix3d R_Right_to_I = Eigen::Matrix3d::Identity();
  R_Right_to_I.block<3, 1>(0, 0) = x_right_inI;
  R_Right_to_I.block<3, 1>(0, 1) = y_right_inI;
  R_Right_to_I.block<3, 1>(0, 2) = z_right_inI;

  Eigen::Matrix3d R_RtoRight = R_Right_to_I.transpose() * R_RtoI;
  cv::Mat mapx_Right_in_R, mapy_Right_in_R;
  create_virt_map(camera_intrinsics.at(1), R_RtoRight, mapx_Right_in_R, mapy_Right_in_R);
  VirtCamMap right_in_R;
  right_in_R.ros_topic = std::string("virt_right/left/image_raw");
  right_in_R.mapx = mapx_Right_in_R;
  right_in_R.mapy = mapy_Right_in_R;
  right_in_R.img_pub = nh->advertise<sensor_msgs::Image>(right_in_R.ros_topic, 2);
  right_in_R.cam_info_topic = std::string("virt_right/left/camera_info");
  right_in_R.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(right_in_R.cam_info_topic, 2);
  left_cam_info.header.frame_id = "depth1";
  right_in_R.cam_info = left_cam_info;
  right_in_R.publish_tf = true;
  Eigen::Quaterniond q_RighttoI(R_Right_to_I);
  quat = tf::Quaternion(q_RighttoI.coeffs()(0), q_RighttoI.coeffs()(1), q_RighttoI.coeffs()(2), q_RighttoI.coeffs()(3));
  Eigen::Vector3d p_RinI = camera_extrinsics.at(1).block(4, 0, 3, 1);
  orig = tf::Vector3(p_RinI.x(), p_RinI.y(), p_RinI.z());
  right_in_R.trans.setOrigin(orig);
  right_in_R.trans.setRotation(quat);
  right_in_R.trans.child_frame_id_ = "depth1";
  right_in_R.trans.frame_id_ = "imu";
  right_in_R.trans.stamp_ = ros::Time::now();
  static_tf.push_back(convert_msg(right_in_R.trans));
  // static_broadcaster.sendTransform(convert_msg(right_in_R.trans));
  virt_cam_map[1].push_back(right_in_R);

  Eigen::Matrix3d R_BRtoI = ov_core::quat_2_Rot(camera_extrinsics.at(2).block(0, 0, 4, 1));
  Eigen::Matrix3d R_BRtoRight = R_Right_to_I.transpose() * R_BRtoI;
  cv::Mat mapx_Right_in_BR, mapy_Right_in_BR;
  create_virt_map(camera_intrinsics.at(2), R_BRtoRight, mapx_Right_in_BR, mapy_Right_in_BR);
  VirtCamMap right_in_BR;
  right_in_BR.ros_topic = std::string("virt_right/right/image_raw");
  right_in_BR.mapx = mapx_Right_in_BR;
  right_in_BR.mapy = mapy_Right_in_BR;
  right_in_BR.img_pub = nh->advertise<sensor_msgs::Image>(right_in_BR.ros_topic, 2);
  right_in_BR.cam_info_topic = std::string("virt_right/right/camera_info");
  right_in_BR.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(right_in_BR.cam_info_topic, 2);
  right_cam_info.P[3] = -right_baseline * VIRT_CAM_F;
  right_cam_info.header.frame_id = "depth1";
  right_in_BR.cam_info = right_cam_info;
  virt_cam_map[2].push_back(right_in_BR);

  Eigen::Vector3d BRtoBL = camera_extrinsics.at(3).block(4, 0, 3, 1) - camera_extrinsics.at(2).block(4, 0, 3, 1);
  double back_baseline = BRtoBL.norm();
  PRINT_DEBUG("back virtual camera baseline: %f\n", back_baseline);
  Eigen::Vector3d x_back_inI = BRtoBL / back_baseline;
  Eigen::Vector3d y_back_inI = -Eigen::Vector3d::UnitY().cross(x_back_inI);
  Eigen::Vector3d z_back_inI = x_back_inI.cross(y_back_inI);

  Eigen::Matrix3d R_Back_to_I = Eigen::Matrix3d::Identity();
  R_Back_to_I.block<3, 1>(0, 0) = x_back_inI;
  R_Back_to_I.block<3, 1>(0, 1) = y_back_inI;
  R_Back_to_I.block<3, 1>(0, 2) = z_back_inI;
  std::cout << "R back to i:\n" << R_Back_to_I << std::endl;

  Eigen::Matrix3d R_BRtoBack = R_Back_to_I.transpose() * R_BRtoI;
  cv::Mat mapx_Back_in_BR, mapy_Back_in_BR;
  create_virt_map(camera_intrinsics.at(2), R_BRtoBack, mapx_Back_in_BR, mapy_Back_in_BR);
  VirtCamMap back_in_BR;
  back_in_BR.ros_topic = std::string("virt_back/left/image_raw");
  back_in_BR.mapx = mapx_Back_in_BR;
  back_in_BR.mapy = mapy_Back_in_BR;
  back_in_BR.img_pub = nh->advertise<sensor_msgs::Image>(back_in_BR.ros_topic, 2);
  back_in_BR.cam_info_topic = std::string("virt_back/left/camera_info");
  back_in_BR.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(back_in_BR.cam_info_topic, 2);
  left_cam_info.header.frame_id = "depth2";
  back_in_BR.cam_info = left_cam_info;
  Eigen::Quaterniond q_BacktoI(R_Back_to_I);
  quat = tf::Quaternion(q_BacktoI.coeffs()(0), q_BacktoI.coeffs()(1), q_BacktoI.coeffs()(2), q_BacktoI.coeffs()(3));
  Eigen::Vector3d p_BRinI = camera_extrinsics.at(2).block(4, 0, 3, 1);
  orig = tf::Vector3(p_BRinI.x(), p_BRinI.y(), p_BRinI.z());
  back_in_BR.trans.setOrigin(orig);
  back_in_BR.trans.setRotation(quat);
  back_in_BR.trans.child_frame_id_ = "depth2";
  back_in_BR.trans.frame_id_ = "imu";
  back_in_BR.trans.stamp_ = ros::Time::now();
  static_tf.push_back(convert_msg(back_in_BR.trans));
  // static_broadcaster.sendTransform(convert_msg(back_in_BR.trans));
  back_in_BR.publish_tf = true;
  virt_cam_map[2].push_back(back_in_BR);

  Eigen::Matrix3d R_BLtoI = ov_core::quat_2_Rot(camera_extrinsics.at(3).block(0, 0, 4, 1));
  Eigen::Matrix3d R_BLtoBack = R_Back_to_I.transpose() * R_BLtoI;
  cv::Mat mapx_Back_in_BL, mapy_Back_in_BL;
  create_virt_map(camera_intrinsics.at(3), R_BLtoBack, mapx_Back_in_BL, mapy_Back_in_BL);
  VirtCamMap back_in_BL;
  back_in_BL.ros_topic = std::string("virt_back/right/image_raw");
  back_in_BL.mapx = mapx_Back_in_BL;
  back_in_BL.mapy = mapy_Back_in_BL;
  back_in_BL.img_pub = nh->advertise<sensor_msgs::Image>(back_in_BL.ros_topic, 2);
  back_in_BL.cam_info_topic = std::string("virt_back/right/camera_info");
  back_in_BL.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(back_in_BL.cam_info_topic, 2);
  right_cam_info.P[3] = -back_baseline * VIRT_CAM_F;
  right_cam_info.header.frame_id = "depth2";
  back_in_BL.cam_info = right_cam_info;
  virt_cam_map[3].push_back(back_in_BL);

  Eigen::Vector3d BLtoL = camera_extrinsics.at(0).block(4, 0, 3, 1) - camera_extrinsics.at(3).block(4, 0, 3, 1);
  double left_baseline = BLtoL.norm();
  PRINT_DEBUG("left virtual camera baseline: %f\n", left_baseline);
  Eigen::Vector3d x_left_inI = BLtoL / left_baseline;
  Eigen::Vector3d y_left_inI = -Eigen::Vector3d::UnitX().cross(x_left_inI);
  Eigen::Vector3d z_left_inI = x_left_inI.cross(y_left_inI);

  Eigen::Matrix3d R_Left_to_I = Eigen::Matrix3d::Identity();
  R_Left_to_I.block<3, 1>(0, 0) = x_left_inI;
  R_Left_to_I.block<3, 1>(0, 1) = y_left_inI;
  R_Left_to_I.block<3, 1>(0, 2) = z_left_inI;

  Eigen::Matrix3d R_BLtoLeft = R_Left_to_I.transpose() * R_BLtoI;
  cv::Mat mapx_Left_in_BL, mapy_Left_in_BL;
  create_virt_map(camera_intrinsics.at(3), R_BLtoLeft, mapx_Left_in_BL, mapy_Left_in_BL);
  VirtCamMap left_in_BL;
  left_in_BL.ros_topic = std::string("virt_left/left/image_raw");
  left_in_BL.mapx = mapx_Left_in_BL;
  left_in_BL.mapy = mapy_Left_in_BL;
  left_in_BL.img_pub = nh->advertise<sensor_msgs::Image>(left_in_BL.ros_topic, 2);
  left_in_BL.cam_info_topic = std::string("virt_left/left/camera_info");
  left_in_BL.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(left_in_BL.cam_info_topic, 2);
  left_cam_info.header.frame_id = "depth3";
  left_in_BL.cam_info = left_cam_info;
  Eigen::Quaterniond q_LefttoI(R_Left_to_I);
  quat = tf::Quaternion(q_LefttoI.coeffs()(0), q_LefttoI.coeffs()(1), q_LefttoI.coeffs()(2), q_LefttoI.coeffs()(3));
  Eigen::Vector3d p_BLinI = camera_extrinsics.at(3).block(4, 0, 3, 1);
  orig = tf::Vector3(p_BLinI.x(), p_BLinI.y(), p_BLinI.z());
  left_in_BL.trans.setOrigin(orig);
  left_in_BL.trans.setRotation(quat);
  left_in_BL.trans.child_frame_id_ = "depth3";
  left_in_BL.trans.frame_id_ = "imu";
  left_in_BL.trans.stamp_ = ros::Time::now();
  static_tf.push_back(convert_msg(left_in_BL.trans));
  // static_broadcaster.sendTransform(convert_msg(left_in_BL.trans));
  left_in_BL.publish_tf = true;
  virt_cam_map[3].push_back(left_in_BL);

  Eigen::Matrix3d R_LtoLeft = R_Left_to_I.transpose() * R_LtoI;
  cv::Mat mapx_Left_in_L, mapy_Left_in_L;
  create_virt_map(camera_intrinsics.at(0), R_LtoLeft, mapx_Left_in_L, mapy_Left_in_L);
  VirtCamMap left_in_L;
  left_in_L.ros_topic = std::string("virt_left/right/image_raw");
  left_in_L.mapx = mapx_Left_in_L;
  left_in_L.mapy = mapy_Left_in_L;
  left_in_L.img_pub = nh->advertise<sensor_msgs::Image>(left_in_L.ros_topic, 2);
  left_in_L.cam_info_topic = std::string("virt_left/right/camera_info");
  left_in_L.cam_info_pub = nh->advertise<sensor_msgs::CameraInfo>(left_in_L.cam_info_topic, 2);
  right_cam_info.P[3] = -left_baseline * VIRT_CAM_F;
  right_cam_info.header.frame_id = "depth3";
  left_in_L.cam_info = right_cam_info;
  virt_cam_map[0].push_back(left_in_L);

  std::vector<ros::Subscriber> sub_vec;
  for(int i = 0; i < num_cameras; i++)
  {
    ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>(cam_topics[i], 1, boost::bind(&img_cbk, i, _1));
    sub_vec.push_back(sub);
  }
  
  ros::spin();
  return 0;
}
