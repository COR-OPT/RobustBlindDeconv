#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def setup_matplotlib():
    matplotlib.use("pgf")
    # when none, svg.fonttype will fall back to the default font used when
    # embedding the image in LaTeX
    plt.rcParams['svg.fonttype'] = "none"
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams.update({
        "text.usetex" : True,
        "pgf.rcfonts" : False,
        "font.size" : 22,
        "text.latex.preamble": [r"\usepackage{times}"]
    })


def phase_plot(save_name, path):
    df = np.loadtxt(path, delimiter=',', skiprows=1)
    ynum = len(np.unique(df[:, 0]))
    xnum = len(np.unique(df[:, 1]))
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(df[:, 3], (ynum, xnum)),
        cmap="gray", interpolation="nearest")
    ax.set_title(r"$ d = 50 $")
    ax.set_xlabel(r"$ \mu_h^2 $")
    ax.set_ylabel(r"$ \frac{m}{2d} $", rotation=0, labelpad=20)
    yidx    = np.arange(ynum)
    ylabels = list(map(lambda x: "$ %s " % format(x), np.arange(ynum) + 1))
    xlabels = [""] * xnum
    xlabels = list(map(lambda x: "$ %.0f $" % x, df[:, 1]))
    # xlabels[0] = "$ {} $".format(int(0.05 * 100))
    # xlabels[-1] = "$ {} $".format(int(1.0 * 100))
    plt.setp(ax, yticks=yidx, yticklabels=ylabels,
        xticks=np.arange(xnum), xticklabels=xlabels)
    plt.savefig(save_name, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Generates a phase transition plot given the phase transition data in a
        pair of .CSV files""")
    parser.add_argument('-i','--in_file',
        help='The input file to read from',
        type=str,
        required=True)
    parser.add_argument("--out_file", '-o',
        type=str,
        help="The path of the output file")
    args = vars(parser.parse_args())
    in_path, out_path = args["in_file"], args["out_file"]
    setup_matplotlib()
    phase_plot(out_path, in_path)
