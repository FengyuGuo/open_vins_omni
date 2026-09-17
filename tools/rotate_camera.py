#!/usr/bin/env python3
"""将 Insta360 相机绕各自的局部 x 轴旋转 180 度。"""

import re
from pathlib import Path

import numpy as np


YAML_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "insta360"
    / "kalibr_imucam_chain.yaml"
)

# T_cam_imu 的连续四行矩阵。
TRANSFORM_RE = re.compile(
    r"(?P<header>^[ \t]*T_cam_imu:[^\n]*\n)"
    r"(?P<rows>(?:^[ \t]*-[ \t]*\[[^]\n]+\][^\n]*\n?){4})",
    re.MULTILINE,
)


def rotate(match):
    lines = match.group("rows").splitlines()
    matrix = np.array(
        [
            [float(value) for value in re.search(r"\[([^]]+)\]", line).group(1).split(",")]
            for line in lines
        ]
    )

    # 相机局部坐标系绕 x 轴旋转 180 度。
    rotation = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # YAML 存的是 T_cam_imu，先转成 T_imu_cam，在相机局部右乘旋转后再转回来。
    matrix = np.matmul(rotation, matrix)

    rows = []
    for old_line, row in zip(lines, matrix):
        indent = re.match(r"^[ \t]*-[ \t]*", old_line).group(0)
        values = ", ".join(
            "0" if abs(value) < 1e-15 else "{:.16g}".format(value) for value in row
        )
        rows.append("{}[{}]".format(indent, values))

    return match.group("header") + "\n".join(rows) + "\n"


def main():
    content = YAML_PATH.read_text()
    content, count = TRANSFORM_RE.subn(rotate, content)
    if count == 0:
        raise RuntimeError("没有找到 T_cam_imu")

    YAML_PATH.write_text(content)
    print("已将 {} 个相机绕局部 x 轴旋转 180 度: {}".format(count, YAML_PATH))


if __name__ == "__main__":
    main()
