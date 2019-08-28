# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 24/08/19
# @Contact: michealabaho265@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def build_line_plots(values, title, labels=[], yticks=None, xticks=None):
    fig, ax = plt.subplots()
    if xticks != None:
        xloc = ticker.MultipleLocator(base=xticks)
        ax.xaxis.set_major_locator(xloc)
    if yticks != None:
        yloc = ticker.MultipleLocator(base=yticks)
        ax.yaxis.set_major_locator(yloc)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    plt.plot(values)
    return plt