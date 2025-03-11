FROM osrf/ros:jazzy-desktop

# =========================================================
# =========================================================

# Are you are looking for how to use this docker file?
#   - https://docs.openvins.com/dev-docker.html
#   - https://docs.docker.com/get-started/
#   - http://wiki.ros.org/docker/Tutorials/Docker

# =========================================================
# =========================================================

# Dependencies we use, catkin tools is very good build system
# Also some helper utilities for fast in terminal edits (nano etc)
RUN apt-get update && apt-get install -y libeigen3-dev nano git && \
    apt-get install -y '~nros-jazzy-rqt*' && \
    apt-get install -y ros-jazzy-turtlesim && \
    # Ceres solver install and setup
    apt-get install -y cmake libatlas-base-dev && \
    apt-get install -y libamd3 libbtf2 libcamd3 libccolamd3 libceres4t64 libcholmod5 libcolamd3 \
    libcxsparse4 libgflags-dev libgflags2.2 libgoogle-glog-dev libgoogle-glog0v6t64 libgraphblas-dev \
    libgraphblas7 libklu2 libldl3 libparu0 librbio4 libspex2 libspqr4 libsuitesparse-dev \
    libsuitesparse-mongoose3 libsuitesparseconfig7 libumfpack6 libunwind-dev libunwind8 && \
    apt-get install -y ros-jazzy-tf* && \
    apt-get install -y ros-jazzy-image* && \
    apt-get install -y ros-jazzy-foxglove-bridge
# ceres will switch from local parameterization to manifold from 2.2.0 on
ENV CERES_VERSION="2.1.0rc2"
RUN git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout tags/${CERES_VERSION} && \
    mkdir build && cd build && \
    # problem with CXSparse linking
    cmake .. -DCXSPARSE=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) install && \
    rm -rf ../../ceres-solver

# Seems this has Python 3.10 installed on it...
RUN apt-get update && apt-get install -y python3-dev python3-matplotlib python3-numpy python3-psutil python3-tk

# Install deps needed for clion remote debugging
# https://blog.jetbrains.com/clion/2020/01/using-docker-with-clion/
# RUN sed -i '6i\source "/catkin_ws/install/setup.bash"\' /ros_entrypoint.sh
RUN apt-get update && apt-get install -y ssh build-essential gcc g++ \
    gdb clang cmake rsync tar python3 && apt-get clean

# ssh stuff

# RUN ( \
#     echo 'LogLevel DEBUG2'; \
#     echo 'PermitRootLogin yes'; \
#     echo 'PasswordAuthentication yes'; \
#     echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
#   ) > /etc/ssh/sshd_config_test_clion \
#   && mkdir /run/sshd
# RUN useradd -m user && yes password | passwd user
# RUN usermod -s /bin/bash user
# CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]