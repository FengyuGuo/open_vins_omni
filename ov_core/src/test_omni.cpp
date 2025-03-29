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

#include <cmath>
#include <deque>
#include <sstream>
#include <unistd.h>
#include <vector>
#include <fstream>

#include "cam/CamOmni.h"
#include "utils/print.h"

using namespace ov_core;

//   distortion_coeffs: [ -0.06727585259390413, 0.5842755541669968, 0.0017763603612335093,
//     -0.0010270945769683538, 0.0]
//   intrinsics: [ 3.2002462594632917, 1661.0508677925284, 1661.1166401802136, 524.1186764831461,
//     630.1494276574698]
//   resolution: [ 1088, 1280]

const int IMG_WIDTH = 1088;
const int IMG_HEIGHT = 1280;

// Main function
int main(int argc, char **argv) {

  CamOmni cam(IMG_WIDTH, IMG_HEIGHT);

  Eigen::MatrixXd cam_calib(8, 1);
  cam_calib << 1661.0508677925284, 1661.1166401802136, 524.1186764831461,  630.1494276574698, 
    -0.06727585259390413, 0.5842755541669968, 0.0017763603612335093, -0.0010270945769683538;
  double fx = cam_calib(0), fy = cam_calib(1), cx = cam_calib(2), cy = cam_calib(3);
  double xi = 3.2002462594632917;
  cam.set_xi(xi);
  std::cout << "xi: " << cam.get_xi() << std::endl;
  cam.set_value(cam_calib);
  double u, v;
  Eigen::Vector2d uv, uv_n;
  Eigen::Vector2d uv_undistort, uv_distort;
  while(std::cin >> u >> v)
  {
    uv << u, v;
    uv_undistort = cam.undistort_d(uv);
    double u_n = (u - cx) / fx, v_n = (v - cy) / fy;
    uv_n << u_n, v_n;
    uv_distort = cam.distort_d(uv_n);
    std::cout << "input point: " << uv.transpose() << std::endl;
    std::cout << "normalized point: " << uv_n.transpose() << std::endl;
    std::cout << "after distortion: " << uv_distort.transpose() << std::endl;
    std::cout << "after undistortion: " << uv_undistort.transpose() << std::endl;
  }
  // Done!
  return EXIT_SUCCESS;
}