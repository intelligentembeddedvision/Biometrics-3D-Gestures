import matplotlib.pyplot as plt
from matplotlib import cm
import math


def cloud3show(points, plot_now=True):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if plot_now:
        plt.show()


def imshow(image, plot_now=True, new_figure=True):
    if new_figure:
        plt.figure()
    img = plt.imshow(image, cmap="inferno")
    if plot_now:
        plt.show()


def cloudshow(image, plot_now=True, new_figure=True):
    if new_figure:
        plt.figure()
    plt.scatter(image[:, 0], image[:, 1])
    if plot_now:
        plt.show()


def plot(data, plot_now=True, new_figure=True):
    if new_figure:
        plt.figure()
    plt.plot(data)
    if plot_now:
        plt.show()


def plot3D(data, plot_now=True, new_figure=True):
    if new_figure:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(data[:, 0], data[:, 1], data[:, 2], cmap=cm.coolwarm)
    if plot_now:
        plt.show()


def plotDist(data, plot_now=True, new_figure=True):
    # fig = plt.figure()
    # im = fig.add_subplot(1)
    if new_figure:
        plt.figure()
    img = plt.imshow(data, cmap="inferno")
    if plot_now:
        plt.show()


def plotCentroids(image, regions, plot_now=True):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    if plot_now:
        plt.show()


def showPlots():
    plt.show()


def addAtIndex(list, index, element):
    if index < len(list):
        list[index] = element
    else:
        list.append(element)
    return list
