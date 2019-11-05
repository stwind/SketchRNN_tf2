import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from sklearn.preprocessing import minmax_scale
from IPython.display import clear_output
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def mpl_rc(values=None):
    fs_title, fs_common = 13, 11
    default = {
        "font.family": ["sans-serif"],
        "font.sans-serif": ["Open Sans", "Arial Unicode MS"],
        "font.size": fs_title,
        "figure.figsize": (8, 6),
        "grid.linewidth": 0.5,
        "legend.fontsize": fs_common,
        "legend.frameon": True,
        "legend.framealpha": 0.6,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 1,
        "axes.facecolor": "#fafafa",
        "axes.labelsize": fs_title,
        "axes.titlesize": fs_title,
        "axes.linewidth": 0.5,
        "xtick.labelsize": fs_common,
        "xtick.minor.visible": False,
        "ytick.labelsize": fs_common,
        "figure.titlesize": fs_title,
        "savefig.pad_inches": 0,
        "savefig.bbox": "tight",
        "animation.html": "html5",
    }
    if values:
        default.update(values)
    return default


def plt_show(filename=None, dpi=None, autoscale=None, rect=None):
    if autoscale:
        plt.autoscale(tight=True)
    plt.tight_layout(rect=rect)
    if filename:
        plt.savefig(filename, dpi=dpi)
    plt.show()


def update_progress(progress, bar_length=20):
    """
    from https://www.mikulskibartosz.name/how-to-display-a-progress-bar-in-jupyter-notebook/
    """
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    progress = max(0, min(1, progress))

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    print(text)


def make_mpl_path(strokes):
    vertices = strokes[:, :-1].cumsum(axis=0, dtype=np.float32) * -1
    if len(vertices) > 0:
        (minx, miny), (maxx, maxy) = vertices.min(0), vertices.max(0)
        aspect = (maxy - miny) / (maxx - minx)

        vertices = minmax_scale(vertices, (-1, 1), axis=0)

        if aspect < 1:
            vertices[:, 1] *= aspect
        else:
            vertices[:, 0] *= aspect
    codes = np.roll(Path.LINETO - strokes[:, -1], 1).astype(int)
    return Path(vertices, codes)


def plot_strokes(ax, strokes, ec="black", no_ticks=True):
    patch = ax.add_patch(PathPatch(make_mpl_path(strokes), lw=1, ec=ec, fc="none"))
    if no_ticks:
        ax.set(xticks=[], yticks=[], frame_on=False)
    else:
        ax.set(frame_on=False)
    ax.axis("equal")
    return patch


def to_normal_strokes(strokes):
    l = np.argmax(strokes[:, 4] == 1)
    return strokes[:l, [0, 1, 3]]


def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    return (1.0 - t) * p0 + t * p1


def report_progress(count, block, total):
    sys.stdout.write("{} / {}\r".format(block * count, total))
    sys.stdout.flush()


def download_dataset(data_class, folder="."):
    filename = "{}.npz".format(data_class)
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        url = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{}".format(
            filename
        )
        print("Downloading: {}".format(url))
        urlretrieve(url, filename=path, reporthook=report_progress)
    print("Downloaded: {}".format(path))
