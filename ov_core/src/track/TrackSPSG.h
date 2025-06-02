/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_TRACK_SPSG_H
#define OV_CORE_TRACK_SPSG_H

#include "TrackBase.h"
#include "Eigen/Core"
#include "super_point.h"
#include "super_glue.h"

#define SUPERPOINT_FEATURE_DIM 259

namespace ov_core {

class TrackSPSG : public TrackBase {

public:
  typedef Eigen::Matrix<double, SUPERPOINT_FEATURE_DIM, Eigen::Dynamic> SUPERPOINT_OUTPUT;
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * 
   */
  explicit TrackSPSG(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod, 
                    const std::string& weight_dir, const std::string& config_path, const std::string& superglue_type);

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  void perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0, SUPERPOINT_OUTPUT &feat0, std::vector<size_t> &ids0);

  void perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1,
                                SUPERPOINT_OUTPUT &feat0, SUPERPOINT_OUTPUT &feat1,
                                size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  void perform_matching(std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, SUPERPOINT_OUTPUT &feat0, SUPERPOINT_OUTPUT &feat1, size_t id0, size_t id1, std::vector<cv::DMatch>& matches);

  // Timing variables
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;


  std::unordered_map<size_t, SUPERPOINT_OUTPUT> last_frame_features;

  //superpoint superglue stuff
  std::string weight_dir;
  std::string config_path;
  std::string superglue_type;

  std::shared_ptr<SuperPoint> superpoint;
  std::shared_ptr<SuperGlue> superglue;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_SPSG_H */
