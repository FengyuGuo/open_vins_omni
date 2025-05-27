#ifndef OV_CORE_CAM_OMNI_H
#define OV_CORE_CAM_OMNI_H

#include "CamBase.h"
#include "utils/tic_toc.h"
#include "utils/print.h"
#include "cmath"

namespace ov_core {

class CamOmni : public CamBase{
public:
    CamOmni(int width, int height) : CamBase(width, height) {}
    ~CamOmni() {}

    Eigen::Vector2f undistort_f(const Eigen::Vector2f& uv_dist) override{
        double x, y, z;
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = uv_dist.x();
        mat.at<float>(0, 1) = uv_dist.y();
        mat = mat.reshape(2);
        // TicToc tic;
        // std::cout << "camera matrix: \n" << camera_k_OPENCV << std::endl;
        // std::cout << "camera distortion: \n" << camera_d_OPENCV << std::endl;
        cv::undistortPoints(mat, mat, camera_k_OPENCV, camera_d_OPENCV); //undistort with radtan camera model
        // std::cout << "cv::undistortPoints cost " << tic.toc() << "ms\n";
        // printf("start omni undistortion\n");
        // PRINT_DEBUG("cam intrinsic: %f, %f, %f, %f\n", camera_values(0), camera_values(1), camera_values(2), camera_values(3));
        // tic.tic();
        omni_lift_projective(mat.at<float>(0), mat.at<float>(1), &x, &y, &z);
        // std::cout << "omni lift projective cost " << tic.toc() << "ms\n"; 
        // printf("lift %f, %f to %f, %f, %f\n", uv_dist.x(), uv_dist.y(), x, y, z);
        double x_n = x / z, y_n = y / z;
        // double fx = camera_values(0), fy = camera_values(1), cx = camera_values(2), cy = camera_values(3);
        // double u = x_n * fx + cx;
        // double v = y_n * fy + cy;
        // PRINT_DEBUG("omni undistort from %f, %f -> %f, %f\n", uv_dist.x(), uv_dist.y(), u, v);
        return Eigen::Vector2f(x_n, y_n);
        // return Eigen::Vector2f(u, v);
    }

    Eigen::Vector2f distort_f(const Eigen::Vector2f& uv_norm) override{

        double u_d, v_d;
        space2plane(uv_norm.x(), uv_norm.y(), 1.0, &u_d, &v_d); // omni camera projection
        // printf("omni prjection %f, %f, 1.0 -> %f, %f\n", uv_norm.x(), uv_norm.y(), u_d, v_d);
        Eigen::MatrixXd cam_d = camera_values;

        // Calculate distorted coordinates for radtan distortion
        double r = std::sqrt(u_d * u_d + v_d * v_d);
        double r_2 = r * r;
        double r_4 = r_2 * r_2;
        double x1 = u_d * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * u_d * v_d +
                    cam_d(7) * (r_2 + 2 * u_d * u_d);
        double y1 = v_d * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * v_d * v_d) +
                    2 * cam_d(7) * u_d * v_d;
        double fx = cam_d(0), fy = cam_d(1), cx = cam_d(2), cy = cam_d(3);
        double u = fx * x1 + cx, v = fy * y1 + cy;
        return Eigen::Vector2f(u, v);
    }

    Eigen::Vector2f distort_space(const Eigen::Vector3f& xyz)
    {
        double u_d, v_d;
        space2plane(xyz.x(), xyz.y(), xyz.z(), &u_d, &v_d); // omni camera projection
        // printf("omni prjection %f, %f, 1.0 -> %f, %f\n", uv_norm.x(), uv_norm.y(), u_d, v_d);
        Eigen::MatrixXd cam_d = camera_values;

        // Calculate distorted coordinates for radtan distortion
        double r = std::sqrt(u_d * u_d + v_d * v_d);
        double r_2 = r * r;
        double r_4 = r_2 * r_2;
        double x1 = u_d * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * u_d * v_d +
                    cam_d(7) * (r_2 + 2 * u_d * u_d);
        double y1 = v_d * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * v_d * v_d) +
                    2 * cam_d(7) * u_d * v_d;
        double fx = cam_d(0), fy = cam_d(1), cx = cam_d(2), cy = cam_d(3);
        double u = fx * x1 + cx, v = fy * y1 + cy;
        return Eigen::Vector2f(u, v);
    }

    void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override{
        // printf("omni came jacobian is not emplemented yet!\n");
        // PRINT_DEBUG("jacobian w.r.t distortion is not implemented yet!\n");
        // exit(1);
        H_dz_dzn.resize(2, 2);
        H_dz_dzn.setZero();
        
        double x = uv_norm.x(), y = uv_norm.y();
        double xi = xi_;
        
        double fx = camera_values(0), fy = camera_values(1);
        double k1 = camera_values(4), k2 = camera_values(5), p1 = camera_values(6), p2 = camera_values(7);

        H_dz_dzn(0, 0) = // d z_u / dx
            fx*(-4*p1*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*p1*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + p2*(-2*pow(x, 4)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*pow(x, 3)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 3*pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) - 2*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) - pow(x, 2)*xi*(k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + x*(k1*(-2*pow(x, 3)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2))*(-4*pow(x, 3)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 4*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 4*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)))/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0) + (k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0));
        H_dz_dzn(0, 1) = // d z_u / dy
            fx*(-4*p1*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*p1*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + p2*(-2*pow(x, 3)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) - x*xi*y*(k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + x*(k1*(-2*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2))*(-4*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 4*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 4*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)))/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0));
        H_dz_dzn(1, 0) = // d z_v / dx
            fy*(p1*(-2*pow(x, 3)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*pow(x, 2)*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) - 4*p2*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*p2*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) - x*xi*y*(k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + y*(k1*(-2*pow(x, 3)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2))*(-4*pow(x, 3)*xi/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 4*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 4*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)))/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0));
        H_dz_dzn(1, 1) = // d z_v / dy
            fy*(p1*(-2*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*x*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*x*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) - 2*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) - 4*p2*x*xi*pow(y, 2)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*p2*x/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) - xi*pow(y, 2)*(k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + y*(k1*(-2*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 2*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 2*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2))*(-4*pow(x, 2)*xi*y/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) - 4*xi*pow(y, 3)/(pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 3)*sqrt(pow(x, 2) + pow(y, 2) + 1)) + 4*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)))/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0) + (k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0));

        H_dz_dzeta.resize(2, 8);
        H_dz_dzeta.setZero();

        H_dz_dzeta(0, 0) = // fx
            2*p1*x*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + p2*(pow(x, 3)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + x*(k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0);
        H_dz_dzeta(0, 1) = 0; // fy
        H_dz_dzeta(0, 2) = 1; // cx
        H_dz_dzeta(0, 3) = 0; // cy
        H_dz_dzeta(0, 4) = // k1
            fx*x*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2))/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0);
        H_dz_dzeta(0, 5) = // k2
            fx*x*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2)/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0);
        H_dz_dzeta(0, 6) = // p1
            2*fx*x*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2);
        H_dz_dzeta(0, 7) = // p2
            fx*(pow(x, 3)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2));
        
        H_dz_dzeta(1, 0) = 0; // fx
        H_dz_dzeta(1, 1) = // fy
            p1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + x*pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + 2*p2*x*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + y*(k1*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2)) + k2*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2) + 1)/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0);
        H_dz_dzeta(1, 2) = 0; // cx
        H_dz_dzeta(1, 3) = 1; // cy
        H_dz_dzeta(1, 4) = // k1
            fy*y*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2))/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0);
        H_dz_dzeta(1, 5) = // k2
            fy*y*pow(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2), 2)/(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0);
        H_dz_dzeta(1, 6) = // p1
            fy*(pow(x, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + x*pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2) + pow(y, 2)/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2));
        H_dz_dzeta(1, 7) = // p2
            2*fy*x*y/pow(xi*sqrt(pow(x, 2) + pow(y, 2) + 1) + 1.0, 2);
    }

private:
    /**
     * @brief migrate from kalibr omni camera projection
     * 
     * @param x 
     * @param y 
     * @param z 
     * @param u 
     * @param v 
     */
    void space2plane(double x, double y, double z, double *u, double *v) const {
        double mx_u, my_u, mx_d, my_d;
        // double fx = camera_values(0), fy = camera_values(1), cx = camera_values(2), cy = camera_values(3);
        // Project points to the normalised plane
        z = z + xi_ * sqrt(x * x + y * y + z * z);
        mx_u = x / z;
        my_u = y / z;

        // Apply distortion
        double dx_u, dy_u;
        omni_distortion(mx_u, my_u, &dx_u, &dy_u); // radtan distort
        mx_d = mx_u + dx_u;
        my_d = my_u + dy_u;

        // Apply generalised projection matrix
        // Matlab points start at 1
        *u = mx_d;
        *v = my_d;
    }
    /**
     * @brief migrate from kalibr omni camera projection
     * 
     * @param mx_u 
     * @param my_u 
     * @param dx_u 
     * @param dy_u 
     */
    void omni_distortion(double mx_u, double my_u, double *dx_u,
                                    double *dy_u) const {
        double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
        double k1 = camera_values(4), k2 = camera_values(5), p1 = camera_values(6), p2 = camera_values(7);
        mx2_u = mx_u * mx_u;
        my2_u = my_u * my_u;
        mxy_u = mx_u * my_u;
        rho2_u = mx2_u + my2_u;
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        *dx_u = mx_u * rad_dist_u + 2 * p1 * mxy_u + p2 * (rho2_u + 2 * mx2_u);
        *dy_u = my_u * rad_dist_u + 2 * p2 * mxy_u + p1 * (rho2_u + 2 * my2_u);
    }

    void omni_distortion(double mx_u, double my_u, double *dx_u,
                                    double *dy_u, double *dxdmx, double *dydmx,
                                    double *dxdmy, double *dydmy) const {
        double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
        double k1 = camera_values(4), k2 = camera_values(5), p1 = camera_values(6), p2 = camera_values(7);
        mx2_u = mx_u * mx_u;
        my2_u = my_u * my_u;
        mxy_u = mx_u * my_u;
        rho2_u = mx2_u + my2_u;
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        *dx_u = mx_u * rad_dist_u + 2 * p1 * mxy_u + p2 * (rho2_u + 2 * mx2_u);
        *dy_u = my_u * rad_dist_u + 2 * p2 * mxy_u + p1 * (rho2_u + 2 * my2_u);

        *dxdmx = 1 + rad_dist_u + k1 * 2 * mx2_u + k2 * rho2_u * 4 * mx2_u
            + 2 * p1 * my_u + 6 * p2 * mx_u;
        *dydmx = k1 * 2 * mx_u * my_u + k2 * 4 * rho2_u * mx_u * my_u
            + p1 * 2 * mx_u + 2 * p2 * my_u;
        *dxdmy = *dydmx;
        *dydmy = 1 + rad_dist_u + k1 * 2 * my2_u + k2 * rho2_u * 4 * my2_u
            + 6 * p1 * my_u + 2 * p2 * mx_u;
    }
    /**
     * @brief omni projection from normalized point to 3D ray
     * 
     * @param u normalized x
     * @param v normalized y
     * @param X 
     * @param Y 
     * @param Z 
     */
    void omni_lift_projective(double u, double v, double *X,
                                         double *Y, double *Z) const {
        double mx_u, my_u;
        double rho2_d;

        // double fx = camera_values(0), fy = camera_values(1), cx = camera_values(2), cy = camera_values(3);
        // mx_d = (u - cx) / fx;
        // my_d = (v - cy) / fy;
        
        omni_undistortGN(u, v, &mx_u, &my_u);
        // printf("before undistort %f, %f, after %f, %f\n", mx_d, my_d, mx_u, my_u);
        //std::cout << "lift projective: u: " << mx_u << ", v: " << my_u << std::endl;
        // printf("xi is %f\n", xi_);
        // Obtain a projective ray
        // Reuse variable
        rho2_d = mx_u * mx_u + my_u * my_u;
        *X = mx_u;
        *Y = my_u;
        *Z = 1 - xi_ * (rho2_d + 1) / (xi_ + sqrt(1 + (1 - xi_ * xi_) * rho2_d));

        //std::cout << "lift projective: p: " << *X << ", " << *Y << ", " << *Z << std::endl;
    }
    void omni_undistortGN(double u_d, double v_d, double * u,
                                     double * v) const {
        *u = u_d;
        *v = v_d;

        double ubar = u_d;
        double vbar = v_d;
        const int n = 5;
        Eigen::Matrix2d F;

        double hat_u_d;
        double hat_v_d;

        // void OmniCameraGeometry::distortion(double mx_u, double my_u, 
        // 					  double *dx_u, double *dy_u,
        // 					  double *dxdmx, double *dydmx,
        // 					  double *dxdmy, double *dydmy) const
        for (int i = 0; i < n; i++) {
            omni_distortion(ubar, vbar, &hat_u_d, &hat_v_d, &F(0, 0), &F(1, 0), &F(0, 1),
                    &F(1, 1));

            Eigen::Vector2d e(u_d - ubar - hat_u_d, v_d - vbar - hat_v_d);
            Eigen::Vector2d du = (F.transpose() * F).inverse() * F.transpose() * e;

            ubar += du[0];
            vbar += du[1];

            if (e.dot(e) < 1e-15)
            break;

        }
        *u = ubar;
        *v = vbar;
    }
};

}// namespace ov_core

#endif