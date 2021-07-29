import numpy as onp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.animation import FuncAnimation

import jax.numpy as np
from jax import vmap

## plotting utilities
def plot_fun(fun, lims=(-5, 5), *args, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    vfun = vmap(fun)
    grid = np.linspace(*lims, num=200)
    ax.plot(grid, vfun(grid), *args, **kwargs)


def equalize_xy_axes(ax):
    """input: matplotlib axis object. sets x and y axis to same limits (and returns new limits)."""
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    lim = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    return lim

def equalize_axes(axs):
    """Argument: list of matplotlib axis objects. Sets x resp. y intervals to the maximum along the list."""
    def transpose_min_max(lims):
        lims = list(zip(*lims))
        lim = (min(lims[0]), max(lims[1]))
        return lim
    xlims = [ax.get_xlim() for ax in axs]
    ylims = [ax.get_ylim() for ax in axs]
    xlim = transpose_min_max(xlims)
    ylim = transpose_min_max(ylims)

    for ax in axs:
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
    return xlim, ylim

def bivariate_hist(xout):
    """argument: np.array of shape (n, 2).
    plots one scatterplot and two heatmaps of the data in 2 dimensions."""
    def myplot(x, y, s, bins=1000):
        heatmap, xlim, ylim = onp.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)

#        lim = [min(ylim[0], xlim[0]), max(ylim[-1], xlim[-1])]
#        extent = lim + lim
        extent = [xlim[0], xlim[-1], ylim[0], ylim[-1]]
        return heatmap.T, extent

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 8)
#    plt.axis('equal')

    x = xout[:, 0]
    y = xout[:, 1]

    sigmas = [0, 32, 64]

    for ax, s in zip(axs.flatten(), sigmas):
        ax.set_aspect("equal")
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title("Scatter plot")
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title("Smoothing with  $\sigma$ = %d" % s)
        equalize_xy_axes(ax)
#        ax.set_aspect("equal")
    plt.show()


def plotobject(data, colors=None, titles=None, xscale="linear", yscale="linear", xlabel=None, ylabel=None, style="-", xaxis=None, axhlines=None):
    """
    * if data is a dict, plot every value.
    * if data is an array, iterate over first axis and plot
    """
    assert type(data) is dict or data.ndim <= 3
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sq = int(np.sqrt(len(data)))
    w = sq + 3
    h = sq
    plt.figure(figsize = [6*w, 2.5*h + 0.2*(h-1)]) # 0.2 = hspace
    if type(data) is dict:
        for i, (k, v) in enumerate(sorted(data.items())):
            plt.subplot(f"{h}{w}{i+1}")
            plt.plot(v, style, color=colors[i])
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.title(k)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
    else:
        for i, v in enumerate(data):
            plt.subplot(f"{h}{w}{i+1}")
            if xaxis is None:
                plt.plot(v, style, color=colors[i])
            else:
                plt.plot(xaxis, v, style, color=colors[i])
            plt.title(titles[i])
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if axhlines is not None:
                plt.axhline(y=axhlines[i], color="y")


def svgd_log(log, style="-", full=False):
    """plot metrics logged during SVGD run."""
    # plot mean and var
    titles = log["metric_names"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors + colors + colors # avoid index out of bound
    for key, dic in log.items():
        if key == "desc":
            plotobject(dic, colors, style=style, xlabel="step")
            colors = colors[len(dic):]

        elif key == "metrics" and full:
            for k, v in dic.items():
                v = np.moveaxis(v, 0, 1)
                plotobject(v, colors, titles[k], yscale="log", style=style) # moveaxis swaps axes 0 and 1
                colors = colors[len(v):]


#@jit(static_argnums=0)
def make_meshgrid(func, lims=(-5, 5), xlims=None, ylims=None, num=100):
    """
    Utility to help with plotting a 2d function.
    Arguments:
    * func: callable. Takes an np.array of shape (2,) as only input, and outputs a scalar.
    * lims
    * num
    Returns:
    meshgrids xx, yy, f(xx, yy)
    """
    if xlims is None:
        xlims = lims
    if ylims is None:
        ylims = lims

    x = np.linspace(*xlims, num)
    y = np.linspace(*ylims, num)
    xx, yy = np.meshgrid(x, y)

    grid = np.stack([xx, yy]) # shape (2, r, r)
    zz = vmap(vmap(func, 1), 1)(grid)
    return xx, yy, zz


def plot_3d(x, y, z, ax=None, **kwargs):
    """makes a 3d plot.
    Arguments:
    * x, y, z are np.arrays of shape (k, k). meant to be output of make_meshgrid."""
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca(projection='3d')
    else:
        ax = ax#(projection='3d')
    ax.plot_surface(x, y, z,
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=True,
                    **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return ax

def plot_fun_2d(pdf, lims=(-5, 5), xlims=None, ylims=None, type="colormesh", num_gridpoints=150, ax=None, cmap="Oranges", **kwargs):
    """
    Arguments
    * pdf: callable, computes a distribution on R2. (ie takes a (2,) array as input
    and returns a scalar)
    * lims: list of two floats (limits)
    * type: string, one of "3d", "contour".
    """
    if xlims is None:
        xlims = lims
    if ylims is None:
        ylims = lims
    if ax is None:
        ax = plt.gca()

    meshgrid = make_meshgrid(pdf, xlims=xlims, ylims=ylims, num=num_gridpoints)
    if type=="3d":
        return plot_3d(*meshgrid, ax=ax, **kwargs)
    elif type=="contour":
        return ax.contour(*meshgrid, **kwargs)
    elif type=="colormesh":
        return ax.pcolormesh(*meshgrid, cmap=cmap, **kwargs)
    else:
        raise ValueError("type must be one of 'colormesh', '3d', or 'contour'.")


def autolabel_bar_chart(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height < 0.01 or height > 10e4:
            ann = "{:.2E}".format(height)
        else:
            ann = '{:.2f}'.format(height)
        ax.annotate(ann,
                    xy = (rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def make_paired_bar_chart(data, labels=None, figax=None):
    """
    Arguments:
    * data: np.array of shape (2, d), where d is the number of pairs of values.
    Can also be shape (2,), in which case d is assumed to be 1.
    Can also be list of length 2.

    * labels is a list of strings of length d.
    * figax = [fig, ax] objects
    """
    assert len(data)==2
    data = np.array(data)
    if data.ndim == 1:
        d = 1
    elif data.ndim == 2:
        d = data.shape[1]
    else:
        raise ValueError("Data must can't have more than 2 dimensions.")

    x = np.arange(d)  # the label locations
    width = 0.35  # the width of the bars

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    rects1 = ax.bar(x - width/2, data[0], width=width, label='Learned h')
    rects2 = ax.bar(x + width/2, data[1], width=width, label='Fixed h')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    ax.set_xticks(x)

    if labels is not None:
        assert len(labels) == d
        ax.set_xticklabels(labels)

    if d == 1:
        ax.set_xlim(-1, 2)


    ax.legend()
    autolabel_bar_chart(ax, rects1)
    autolabel_bar_chart(ax, rects2)

    fig.tight_layout()

    if figax is None:
        return fig, ax

    return rects1, rects2

def set_axhlines(ax, ys):
    """Paint len(ys) segmented horizontal lines onto plot given by ax."""
    try:
        ns = len(ys)
    except TypeError:
        ys = [ys]
        ns = len(ys)
    grid = np.linspace(0, 1, num=ns+1)

    for i, y in enumerate(ys):
        ax.axhline(y, xmin=grid[i], xmax=grid[i+1], color="k", linestyle="--", linewidth=2.5)

def errorfill(x, y, yerr, color="r", alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def scatter(x, *args, ax=None, **kwargs):
    if ax is None:
        ax=plt.gca()
    ax.scatter(*np.rollaxis(x, 1), *args, **kwargs)

def quiverplot(f, samples=None, num_gridpoints=50, ax=None, lims=[-10, 10], xlims=None, ylims=None, angles="xy", scale=2, **kwargs):
    """
    Plot a vector field. f is a function f: R^2 ---> R^2
    If arrows are too large, change scale (larger scale = shorter arrows)
    If samples is not None, then draw arrows at samples insted of building a grid.
    In this case, the arguments num_gridpoints, lims, xlims, ylims are ignored.
    """
    if xlims is None:
        xlims = lims
    if ylims is None:
        ylims = lims
    if ax is None:
        ax = plt.gca()

    if samples is None:
        def split_f(x, y):
            return f(np.append(x, y))
        x = np.linspace(*xlims, num_gridpoints)
        y = np.linspace(*ylims, num_gridpoints)
        xx, yy = np.meshgrid(x, y, dtype=np.float32)
        zz = vmap(vmap(split_f))(xx, yy)
        uu, vv = np.rollaxis(zz, 2)
        ax.quiver(xx, yy, uu, vv, angles=angles, scale=scale, **kwargs)
    else:
        zz = vmap(f)(samples)
        u, v = np.rollaxis(zz, 1)
        x, y = np.rollaxis(samples, 1).split(2)
        x, y = np.squeeze(x), np.squeeze(y)
        ax.scatter(x, y, color="black")
        ax.quiver(x, y, u, v, angles=angles, scale=scale, **kwargs)

def animate_array(arr, fig=None, ax=None, interval=100, color=None):
    """Animate array of shape (n_timesteps, n_points, 2)
    as moving scatterplot. Needs `%matplotlib widget` to work in Jupyter lab.
    interval = ms between frames."""
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # plot the first frame
    scat = ax.scatter(arr[0, :, 0], arr[0, :, 1], color=color)

    title = ax.get_title()
    # animation fn
    def animate(i):
        scat.set_offsets(arr[i])
        ax.set_title(title + f": Timestep {i}")
        return scat, ax

    # call animation
    t = len(arr)
    anim = FuncAnimation(fig, animate, interval=interval, frames=t, blit=True)
    return anim

def plot_gradient_field(v: callable, ax=None, samples=None, lims=(-5, 5), color="green", **kwargs):
    """Plot the gradient field v.
    v is a function that maps a (n, 2) batch of points in 2D
    to an (n, 2) batch of vectors."""
    if ax is None:
        ax = plt.gca()

    if samples is not None:
        xx = samples
    else:
        grid = np.linspace(*lims, 25)
        xx = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)

    scores = v(xx)
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # change width to make arrows thicker
    ax.quiver(*xx.T, *scores_log1p.T, width=0.005, color=color, **kwargs)
    return
