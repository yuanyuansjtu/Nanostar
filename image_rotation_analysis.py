import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from PIL import Image
# from scipy.misc import imread
from matplotlib.patches import Circle
import pickle
import os
import glob


# Reference from this code
# https://codereview.stackexchange.com/questions/41688/rotating-greyscale-images

# import numpy as np

def rotate_coords(x, y, theta, ox, oy):
    """Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy).

    """
    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy


def rotate_image(src, theta, ox, oy):
    """Rotate the image src by theta radians about (ox, oy).
    Pixels in the result that don't correspond to pixels in src are
    replaced by the value fill.

    """
    # Images have origin at the top left, so negate the angle.
    theta = -theta
    fill = src.mean(axis=0).mean()

    # Dimensions of source image. Note that scipy.misc.imread loads
    # images in row-major order, so src.shape gives (height, width).
    sh, sw = src.shape

    # Rotated positions of the corners of the source image.
    # cx, cy = rotate_coords([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)

    # Determine dimensions of destination image.
    # dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

    # Coordinates of pixels in destination image.
    # dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
    dx, dy = np.meshgrid(np.arange(sw), np.arange(sh))

    # Corresponding coordinates in source image. Since we are
    # transforming dest-to-src here, the rotation is negated.
    # sx, sy = rotate_coords(dx + cx.min(), dy + cy.min(), -theta, ox, oy)

    sx, sy = rotate_coords(dx, dy, -theta, ox, oy)

    # Select nearest neighbour.
    sx, sy = sx.round().astype(int), sy.round().astype(int)

    # print(dh)

    # print(sx)
    # Mask for valid coordinates.
    mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)
    # print(mask)
    # Create destination image.
    # dest = np.empty(shape=(dh, dw), dtype=src.dtype)
    src_array = np.array(src)

    # dest = np.empty(shape=(dh, dw))
    dest = np.empty(shape=(sh, sw))
    # Copy valid coordinates from source image.
    dest[dy[mask], dx[mask]] = src_array[sy[mask], sx[mask]]

    # Fill invalid coordinates.
    dest[dy[~mask], dx[~mask]] = fill

    return dest


def check_rotation(img, theta, ox, oy, cap_width_pixels, cap_analyze_length):
    dest = rotate_image(img, theta, ox, oy)
    figure, ax = plt.subplots(1, figsize=(8, 8 * 1.33))
    circ1 = Circle((ox, oy), 5)
    circ2 = Circle((ox + cap_width_pixels, oy), 5)
    circ3 = Circle((ox, oy + cap_analyze_length), 5, color='green')
    circ4 = Circle((ox + cap_width_pixels, oy + cap_analyze_length), 5, color='green')
    circ5 = Circle((ox, oy - int(cap_analyze_length / 2)), 5, color='yellow')
    circ6 = Circle((ox + cap_width_pixels, oy - int(cap_analyze_length / 2)), 5, color='yellow')
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.add_patch(circ3)
    ax.add_patch(circ4)
    ax.add_patch(circ5)
    ax.add_patch(circ6)
    plt.imshow(dest, cmap="gray")
    plt.show()
    figure, ax = plt.subplots(1, figsize=(8, 8 * 1.33))
    plt.imshow(dest, cmap="gray")
    plt.show()
    return dest


def check_up_downstream(dest, ox, oy, cap_width_pixels, cap_analyze_length):
    figure, ax = plt.subplots(1, figsize=(8, 8 * 1.33))
    y_location = []
    x_temperatures = []

    for y in np.arange(-int(cap_analyze_length / 2), cap_analyze_length, 10):
        y_location.append(y)
        x_temperatures.append(dest[oy + y, ox:ox + cap_width_pixels])
        plt.plot(dest[oy + y, ox:ox + cap_width_pixels], label='y = ' + str(y))

    plt.legend()
    plt.xlabel('x pixels')
    plt.ylabel('Temperature Degs')
    plt.show()

    df_temperature_along_y = pd.DataFrame(data=np.array(x_temperatures).T, columns=y_location)

    return df_temperature_along_y


def compute_inlet_outlet_temperature(file_name, directory_path,
                                     ox, oy, theta, y_offset_upsteam, y_offset_downsteam, cap_width_pixels):
    file_path = directory_path + 'temperature data//' + file_name + "//"
    dump_file_name = file_name + '.p'
    dump_exist = False

    dump_file_path = directory_path + "temperature dump//" + dump_file_name

    if os.path.isfile(dump_file_path):  # if dump file already exist
        temp_full = pickle.load(open(dump_file_path, 'rb'))
        N_files = np.shape(temp_full)[2]
        dump_exist = True
        print("existing dump file found!")
    else:
        list_file = os.listdir(file_path)  # dir is your directory path
        N_files = len(list_file)

        arr = glob.glob(file_path + '*.csv')
        arr = sorted(arr, key=lambda x: int(x.split(file_name + '_')[1].split('.')[0]))

        shape_single_frame = pd.read_csv(arr[0], header=None, skiprows=5).shape  # frame 0 must be there
        temp_full = np.zeros((shape_single_frame[0], shape_single_frame[1], N_files))

        # arr = os.listdir(file_path)
        for idx, file_name_ in enumerate(arr):
            temp = pd.read_csv(file_name_, header=None, skiprows=5).values.tolist()
            if np.shape(temp)[0] == 480 and np.shape(temp)[1] == 640:
                temp_full[:, :, idx] = temp

            else:
                print('The input file at index ' + str(idx) + ' is illegal!')
        pickle.dump(temp_full, open(dump_file_path, "wb"))

    temp_inlet = np.zeros(temp_full.shape[2])
    temp_outlet = np.zeros(temp_full.shape[2])

    for idx in range(N_files):
        rotated_image = rotate_image(temp_full[:, :, idx], theta, ox, oy)
        if (idx > 0 and idx % 100 == 0):
            figure, ax = plt.subplots(1, figsize=(8, 8 * 1.33))
            circ1 = Circle((ox, oy), 5)
            circ2 = Circle((ox + cap_width_pixels, oy), 5)
            circ3 = Circle((ox, oy + y_offset_downsteam), 5, color='green')
            circ4 = Circle((ox + cap_width_pixels, oy + y_offset_downsteam), 5, color='green')
            circ5 = Circle((ox, oy + y_offset_upsteam), 5, color='yellow')
            circ6 = Circle((ox + cap_width_pixels, oy + y_offset_upsteam), 5, color='yellow')
            ax.add_patch(circ1)
            ax.add_patch(circ2)
            ax.add_patch(circ3)
            ax.add_patch(circ4)
            ax.add_patch(circ5)
            ax.add_patch(circ6)
            plt.imshow(rotated_image, cmap="gray")
            plt.show()

        temp_inlet[idx] = rotated_image[oy + y_offset_upsteam, ox:ox + cap_width_pixels].mean()
        temp_outlet[idx] = rotated_image[oy + y_offset_downsteam, ox:ox + cap_width_pixels].mean()

    return temp_inlet, temp_outlet


def plot_temperature_profile(temp_inlet, temp_outlet, frame_skip, frame_steady_start, frame_steady_end):
    f = plt.figure(figsize=(17, 5))
    plt.subplot(131)
    plt.plot(temp_inlet, label='inlet temperature')
    plt.plot(temp_outlet, label='outlet temperature')
    plt.xlabel('# of tempearture measurements', fontsize='10', fontweight='bold')
    plt.ylabel('temperature Degs', fontsize='10', fontweight='bold')
    plt.title('inlet & outlet temperature', fontsize='10', fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 8})
    # plt.legend(prop = {'weight':'bold','size': 12})

    plt.subplot(132)
    frame_steady_start_ = int(frame_steady_start / (1 + frame_skip))
    frame_steady_end_ = int(frame_steady_end / (1 + frame_skip))
    frame_index = np.arange(frame_steady_start_, frame_steady_end_)
    plt.plot(temp_inlet[frame_index], label='inlet temperature')
    plt.plot(temp_outlet[frame_index], label='outlet temperature')
    plt.xlabel('# of tempearture measurements', fontsize='10', fontweight='bold')
    plt.ylabel('temperature Degs', fontsize='10', fontweight='bold')
    plt.title('inlet & outlet steady temperature', fontsize='10', fontweight='bold')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 8})

    plt.legend(prop={'weight': 'bold', 'size': 12})
    # plt.show()

    plt.subplot(133)
    plt.plot(temp_outlet[frame_index] - temp_inlet[frame_index])
    plt.xlabel('# of tempearture measurements', fontsize='10', fontweight='bold')
    plt.ylabel('temperature Degs', fontsize='10', fontweight='bold')
    plt.title('temperature rise', fontsize='10', fontweight='bold')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    return np.mean(temp_outlet[frame_index] - temp_inlet[frame_index])


def cal_laser_absorption(tube_width, flow_rate, temp_rise):  # flow rate in ul/min
    Aw = (tube_width - 0.07 / 1000 * 2) * 0.1 / 1000  # unit m2
    Cp_water = 4200
    rho_water = 998
    mdot = flow_rate * 10 ** (-6) / 1000 / 60 * rho_water
    P = mdot * Cp_water * temp_rise
    return P


def check_image(directory_path, file_name, frame_show, theta, ox, oy, cap_width_pixels, cap_analyze_length):
    # note please let csv files include time stamp data in the top 5 rows
    img = pd.read_csv(
        directory_path + 'temperature data//' + file_name + '//' + file_name + '_' + str(frame_show) + '.csv', sep=',',
        header=None, skiprows=5)

    print("Original image shown here.--------------------------------------------------")
    plt.figure(figsize=(8, 8 * 1.33))
    plt.imshow(img, cmap="gray")
    plt.show()

    print("Rotated image shown here.--------------------------------------------------")
    dest = check_rotation(img, theta, ox, oy, cap_width_pixels, cap_analyze_length)
    down_stream_temperature = check_up_downstream(dest, ox, oy, cap_width_pixels, cap_analyze_length)

    down_stream_temperature.to_csv("{}_avg_temperature_demonstration.csv".format(file_name))


def compute_heating_power(file_name, directory_path, frame_skip, ox, oy, theta, y_offset_upsteam, y_offset_downsteam,
                          cap_width_pixels, frame_steady_start, frame_steady_end, cap_width_real_dimension, flow_rate):
    # frame_skip = 0

    inlet_temp, outlet_temp = compute_inlet_outlet_temperature(file_name, directory_path,
                                                               ox, oy, theta, y_offset_upsteam, y_offset_downsteam,
                                                               cap_width_pixels)

    T_rise = plot_temperature_profile(inlet_temp, outlet_temp, frame_skip, frame_steady_start, frame_steady_end)
    print('The mean temperature rise is ' + "{0:.2f}".format(T_rise) + 'K')

    P = cal_laser_absorption(cap_width_real_dimension, flow_rate, T_rise)  # flow rate in ul/min
    print('The power of absorbed laser is: ' + "{0:.4f}".format(P) + 'W.')

    return inlet_temp, outlet_temp
