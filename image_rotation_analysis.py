import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from PIL import Image
# from scipy.misc import imread
from matplotlib.patches import Circle

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

def check_rotation(img,theta,ox,oy,cap_width,cap_analyze_length):
    dest = rotate_image(img,theta,ox,oy)
    figure,ax = plt.subplots(1,figsize=(6,8))
    circ1 = Circle((ox,oy),5)
    circ2 = Circle((ox+cap_width,oy),5)
    circ3 = Circle((ox,oy+cap_analyze_length),5,color = 'green')
    circ4 = Circle((ox+cap_width,oy+cap_analyze_length),5,color = 'green')
    circ5 = Circle((ox,oy-int(cap_analyze_length/2)),5,color = 'yellow')
    circ6 = Circle((ox+cap_width,oy-int(cap_analyze_length/2)),5,color = 'yellow')
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.add_patch(circ3)
    ax.add_patch(circ4)
    ax.add_patch(circ5)
    ax.add_patch(circ6)
    plt.imshow(dest, cmap="gray")
    plt.show()

def check_up_downstream(dest,ox,oy,cap_width,cap_analyze_length):
    for y in np.arange(-int(cap_analyze_length/2),cap_analyze_length,10):
        plt.plot(dest[oy+y,ox:ox+cap_width],label = 'y = '+str(y))
    plt.legend()
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.show()

