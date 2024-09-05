import json
import os

import numpy as np
from matplotlib import pyplot as plt


def create_table(src_dir):
    rows = []
    x = []
    y = []

    roll = []
    pitch = []
    yaw = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    # print(os.path.join(root, file))
                    data = json.load(f)
                    rate = data['stats']['detection_rate'] * 100
                    dist = data['stats']['avg_distance'][0] * 1000
                    pose = ", ".join([f"{p*1000:.2f}" for p in data['stats']['avg_position'][0]])
                    ori = ", ".join([f"{o:.2f}" for o in data['stats']['avg_abs_orientation'][0]])
                    gt_dist = int(data['args']['note'].split('mm')[0])
                    ref = "_".join(root.split('/')[-1].split('_')[0:2])
                    dist_err = 100 * abs(dist - gt_dist) / gt_dist

                    roll_err = abs(180 - data['stats']['avg_abs_orientation'][0][0])
                    pitch_err = abs(data['stats']['avg_abs_orientation'][0][1])
                    yaw_err = abs(data['stats']['avg_abs_orientation'][0][2])

                    rows.append(f"{gt_dist}\t{rate:.2f}\t{dist:.2f} ({dist_err:.2f}%)\t{pose}\t{ori}\t{ref}")
                    x.append(gt_dist)
                    y.append(dist_err)
                    roll.append(roll_err)
                    pitch.append(pitch_err)
                    yaw.append(yaw_err)

    for row in sorted(rows):
        print(row)

    compute_quadratic(x, y, 'dist')
    compute_quadratic(x, roll, 'roll')
    compute_quadratic(x, pitch, 'pitch')
    compute_quadratic(x, yaw, 'yaw')


def quadratic_function(x, coefficients):
    a, b, c = coefficients
    return a * x ** 2 + b * x + c


def compute_quadratic(x, y, name):
    paired_list = list(zip(x, y))
    paired_list.sort()
    sorted_x, reordered_y = zip(*paired_list)

    x = list(sorted_x)
    y = list(reordered_y)

    print(x)
    print(y)
    error_coefficients = np.polyfit(x, y, 2)
    print(error_coefficients)
    # print(quadratic_function(60, error_coefficients))
    # print(quadratic_function(75, error_coefficients))
    # print(quadratic_function(80, error_coefficients))

    x1 = np.arange(60, 126, 0.1)
    y1 = quadratic_function(x1, error_coefficients)

    plt.plot(x, y, '-o')
    plt.plot(x1, y1)
    x_ticks = np.arange(60, 130, 5)
    plt.xticks(x_ticks)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Error °")
    # plt.ylabel("% Error")
    # plt.show()
    # plt.savefig(f'aug6/quadratic_function_{name}.png', dpi=300)


if __name__ == '__main__':
    create_table("./aug5")