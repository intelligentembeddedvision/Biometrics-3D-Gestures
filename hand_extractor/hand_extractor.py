# ratio threshold from 1.5 to 1.45
# MIN_POINTS for bbClosestPoint from 200 to 300
from scipy import ndimage

from hand_extractor.visualizer import *
from hand_extractor.utilities import *

from skimage.measure import regionprops, label
from sklearn.decomposition import PCA

from copy import deepcopy

import pickle
from time import time

from main import load_data

SHOW_IMAGES = True
CLI_VERBOSE = True


def hd_filter(dstImage):
    result = ndimage.median_filter(dstImage, size=3)

    return result


def hd_getClosestPoint(filteredImage):
    # dim = size(filter_dist)
    HAND_DEPTH = 0.08

    MIN_POINTS = 300
    sorted_array = np.sort(filteredImage.flatten())

    flag = 0
    for pct in sorted_array:
        handMask = (filteredImage > pct) * (filteredImage < (pct + HAND_DEPTH))
        if (handMask.sum() >= MIN_POINTS):
            hand = (filteredImage - pct) * handMask
            return (hand, flag)
        else:
            flag = flag + 1
    return ([], flag)


def hd_rectangle(data):
    # mask with the non-zero points
    zero_mask = data > 0
    # doing a sum on the first dimension will result in a vector indication
    # with non zero values the non-zero columns
    non_zero_columns = np.sum(zero_mask, 0)
    # doing a sum on the second dimension will result in a vector indication
    # with non zero values the non-zero rows
    non_zero_rows = np.sum(zero_mask, 1)

    # np.where returns a vector with the indices of the elements satisfying
    # the rule
    non_zero_columns = np.where(non_zero_columns > 0)
    non_zero_rows = np.where(non_zero_rows > 0)

    # extracting the bounding rows and columns
    # the np.where returns a tuple of arrays, that in our case contains
    # just one array with the indices we are interested in
    # so we must extract that array first using index 0 -> [0]
    # and only then we can extract our elements
    non_zero_columns = non_zero_columns[0]
    non_zero_rows = non_zero_rows[0]

    # because of how numpy works, we cannot extract a region using two
    # vectors of indices, so we extract the rows and then the columns
    return data[non_zero_rows, :][:, non_zero_columns]


def hd_extract_centroids(image):
    image_labels = label(image)
    regions = regionprops(image_labels)
    return regions


def hd_extract_regions_of_interest(sample):
    MIN_AREA = 80
    # multiply by 1 to convert from boolean to float values
    sample_binarized = 1 * (sample > 0)

    regions = hd_extract_centroids(sample_binarized)

    filtered_regions = []
    if len(regions) == 1:
        filtered_regions.append(regions[0])
    elif len(regions) > 1:
        for region in regions:
            if region.area > MIN_AREA:
                filtered_regions.append(region)
        if len(filtered_regions) > 1:
            regions = filtered_regions
            filtered_regions = []
            for region in regions:
                if region.centroid[0] < sample_binarized.shape[0] / 2:
                    filtered_regions.append(region)
        if len(filtered_regions) > 1:
            regions = filtered_regions
            filtered_regions = []
            for region in regions:
                if region.centroid[1] < sample_binarized.shape[1] / 2:
                    filtered_regions.append(region)

    return filtered_regions


def hd_take_out_zeros(image):
    point_list = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] > 0):
                point_list.append([i, j, image[i][j]])
    return np.array(point_list)


def hd_remove_forearm(hand):
    # python passes lists by reference and we do not want to modify
    # the lists outside of the function
    # we use deepcopy to create a copy that doesn't have references to the original
    # this is done for modularity, but could be discarded as an optimisation
    hand = deepcopy(hand)
    data = hd_take_out_zeros(hand)
    # we only need the coordinates
    data = deepcopy(data[:, 0:2])

    # PCA analysis on the data pints
    pca = PCA(n_components=2)
    # pca.fit_transform returns the transformed data ( the y from the previous code)
    transformed_data = pca.fit_transform(data)
    x = pca.components_

    # we must check if the data has been flipped on the vertical axis
    # otherwise, during segmentation we will start segmenting from the forearm
    # instead of starting from the hand
    if x[0, 0] < 0:
        x[0, 0] = -x[0, 0]
        transformed_data[:, 0] = - transformed_data[:, 0]

    if x[1, 1] < 0:
        x[1, 1] = -x[1, 1]
        transformed_data[:, 1] = - transformed_data[:, 1]

    # the offset between the center of the original data set and the transformed data
    # is going to be needed later when forearm is going to be removed
    # the offset can be retrieved from the pca instance
    absolute_offset = pca.mean_

    # the data is reordered so the points are in ascending order with respect
    # to the x axis
    #
    # transformed_data contains x and y on its colums [x,y]
    # we want to sort the points by their x coordinate
    # using the 'sort' function, it will sort both colums independetly (not what we want)
    # instead, we use argsort for the first colums, to optain the indexis of
    # the sorted array and than we use those to reorder the points
    ordered_indexes = transformed_data[:, 0].argsort()
    transformed_data = transformed_data[ordered_indexes]

    y = transformed_data
    if CLI_VERBOSE:
        print(x)

    # the ratio between the maximum deviations on the x and y axes is calculated
    # to quantify the elongation of the shape
    # if the foream is present in the selection, the elongation is going to be big
    ratio = np.abs(y[:, 0]).max() / np.abs(y[:, 1]).max()

    # if the elongation is smaller than a threshold (determined by experimentation)
    # the segmentation process is stopped
    if CLI_VERBOSE:
        print(ratio)
    # modified from 1.5 because of differences in floating values calculated
    # in python vs matlab
    ## possible improvement by using a kenel centered on the centroid
    ## and calculating ratio between the inside and outside
    ## if there is no forearm, the kernel should be mostly centered on the hand,
    ## and most of the points should be on the inside
    ### this would also solve the problem with the flipped data (x[1,1])
    if ratio < 1.45:
        if CLI_VERBOSE:
            print('Ratio too small. Segmentation stopped!')
        return (hand > 0)

    # the sizes of the searching boxes are initialised
    dataSize = transformed_data.shape[0]
    track = []
    trackIndex = []
    ftrack = []
    initialObj1Size = 300
    forearmWindow = 100

    # if the dataset is too small (smaller than the minimum size) the segmentation process is stopped
    if dataSize < initialObj1Size + forearmWindow + 1:
        return (hand > 0)

    # the size of the searching box is modified (increased) every step and a new
    # segmentation value is calculated
    # 'track' will hold the values used latter in decideing the point of segmentation
    # 'trackIndex' will store the indexes at which the values in the 'track' array have been calculated
    ####################################################################################################################
    ## possible speed improvement by using the cumulative sum
    obj1 = np.sum(np.abs(y[0: initialObj1Size - 1, 1]))
    obj2 = np.sum(np.abs(y[initialObj1Size - 1: (initialObj1Size - 1) + forearmWindow, 1]))
    for i in range(initialObj1Size, dataSize - (forearmWindow + 1)):
        obj1 = obj1 + np.abs(y[i, 1])
        obj2 = obj2 + np.abs(y[i + forearmWindow, 1]) - np.abs(y[i - 1, 1])

        trackIndex.append(i)
        track.append((obj2 / forearmWindow) / (obj1 / i))

    ## possible improvement by using an actual filtering function
    ## or by doing the moving average while the ratio is calculated
    ## (the for loop above)
    # filter to remove the high frequency noise
    fsize = 50
    ftrack = []
    ftrackIndex = []
    for j in range(0, len(track) - fsize):
        ftrack.append(np.mean(track[j:j + fsize]))
        ftrackIndex.append(trackIndex[j + fsize])
    # filter to remove/attenuate the variation
    fsize2 = 90
    ftrack2 = []
    ftrackIndex2 = []
    for j in range(0, len(ftrack) - fsize2):
        ftrack2.append(np.mean(ftrack[j:j + fsize2]))
        ftrackIndex2.append(ftrackIndex[j + fsize2])

    variation = np.array(ftrack[fsize2:]) - np.array(ftrack2)
    # filtering the variation to remove the small high frequency
    # noise left
    fsize3 = 70
    ftrack3 = []
    ftrackIndex3 = []
    for j in range(0, len(variation) - fsize3):
        ftrack3.append(np.mean(variation[j:j + fsize3]))
        ftrackIndex3.append(ftrackIndex2[j + fsize3])
    # variation = np.array(ftrack[fsize2:]) - np.array(ftrack2)
    if SHOW_IMAGES:
        plot(ftrack[fsize2 + fsize3:], False)
    ftrack = np.array(ftrack3) * 4 + np.array(ftrack2[fsize3:])
    ftrackIndex = ftrackIndex3

    if SHOW_IMAGES:
        plot(ftrack, False, False)

    # the segmentation point is going to be determinated by analizeing the data
    # from the segmentation function above
    #
    # 'strackDir' stores 1 or -1 depending on whether the function is increasing
    # or decreasing ( 1 is for increasing and -1 for decreasing)
    #
    # 'strack' -> strack[i] stores for how many points the funciton was increasing
    # or decreasing depending on strackDir[i]
    #
    # 'strackIndex' stores the last index (form the original data) at which the function
    # was still increasing (strackDir[i]==1) or decreasing (strackDir[i]==-1)

    ## I am sure this can be done more efficiently using numpy functions
    ## and some optimisations
    ## but I didn't have enough time to figure it out
    strackDir = []
    if ftrack[1] >= ftrack[0]:
        strackDir.append(1)
    else:
        strackDir.append(-1)
    strack = [1]
    strackIndex = [ftrackIndex[0]]

    for i in range(1, len(ftrack) - 1):
        if ftrack[i + 1] == ftrack[i]:
            strack[-1] = strack[-1] + 1
            strackIndex[-1] = ftrackIndex[i]
        else:
            sign = -1
            if ftrack[i + 1] >= ftrack[i]:
                sign = 1
            if strackDir[-1] == sign:
                strack[-1] = strack[-1] + 1
                strackIndex[-1] = ftrackIndex[i]
            else:
                strackDir.append(sign)
                strack.append(1)
                strackIndex.append(ftrackIndex[i])

    # the information regarding the monotonicity of the function is compressed
    # by eliminating the short changes in monotonicity
    # if the longest range without a change in monotonicity is N samples
    # a short change has fewer than 0.03*N samples
    threshold = np.max(strack) * 0.03

    j = 0
    while j < len(strack):
        if strack[j] < threshold:
            strack.pop(j)
            strackDir.pop(j)
            strackIndex.pop(j)
            if (j > 0) and (j < len(strack)):
                if strackDir[j] == strackDir[j - 1]:
                    strack[j] = strack[j] + strack[j - 1]
                    strack.pop(j - 1)
                    strackDir.pop(j - 1)
                    strackIndex.pop(j - 1)
        else:
            j = j + 1

    # if the monotonicity of the funciton doesn't change, the segmentation is stopped
    if len(strack) == 1:
        if CLI_VERBOSE:
            print('No monotonicity changes deteced. Segmentation stopped!')
        return (hand > 0)

    # implementing the algorithm from fig.9 on page 7
    # the arrays are converted to numpy arrays to benefit from some usefull functionaloty
    # of this datatype
    ## observation after implementation: ftrack, ftrackIndex, strack, strackIndex 
    ## are used further, but only ftrack, ftruckIndex whould benefit from being numpy arrays
    ## leaving the others if usefull in future development - can be removed later
    ftrack = np.array(ftrack)
    ftrackIndex = np.array(ftrackIndex)
    strack = np.array(strack)
    strackIndex = np.array(strackIndex)
    strackDir = np.array(strackDir)
    minimum = np.max(ftrack)
    minimum1 = np.max(ftrack)
    k = 0
    k1 = 0

    flag = False
    for j in range(0, len(strack)):
        if strackDir[j] == -1:
            if not flag:
                flag = True
                minimum = ftrack[ftrackIndex == strackIndex[j]]
                k = j
                minimum1 = minimum
                k1 = k
            else:
                if ftrack[ftrackIndex == strackIndex[j]] < minimum:
                    minimum = ftrack[ftrackIndex == strackIndex[j]]
                    k = j

    if CLI_VERBOSE:
        print(minimum)
        print(minimum1)
        print(minimum / minimum1)
        print(np.where(ftrack == minimum))
        print(np.where(ftrack == minimum1))

    if minimum / minimum1 > 0.8:
        minimum = minimum1
        k = k1

    if CLI_VERBOSE:
        print(minimum)
        print(np.where(ftrack == minimum))

    if k == len(strack) - 1:
        if len(strack) > 2 and (
                np.where(ftrackIndex == strackIndex[k - 2])[0][0] / np.where(ftrackIndex == strackIndex[k])[0][
            0] > 0.7):
            k = k - 2
            minimum = ftrack[ftrackIndex == strackIndex[k]]
        else:
            if CLI_VERBOSE:
                print('Global minimum at end of selection. Segmentation stopped!')
            return (hand > 0)

    segIndex = strackIndex[k]
    #
    # %     se vizualizeaza functia de segmentare si decizia de segmentare luata
    # %     h = figure(3), set(h,'OuterPosition',[900 500 400 400]);
    # %     plot(ftrackIndex,ftrack,[segIndex segIndex],[0.95*minimum 1.05*minimum],'LineWidth',1.5);
    # %     title(['Segmentation Index:' num2str(segIndex)]);
    # %     xlabel('Point Index');
    # %     ylabel('Decision Function');

    relativeOffset = np.round(y[segIndex, 0])
    offset = absolute_offset + np.dot(relativeOffset, np.array(x[:, 0]).T)
    # print(image.shape)
    # print(hand.shape)
    # because the coordinates are 1x2, multiplying with x (2x2) will result in a 1x2 matrix
    # only the the x coordinate of the resulting point (first element of the output matrix) 
    # is relevant, so we can eliminate the second column of x to simplify the comparison later 
    c1 = x[0, 0]
    ## this will improve segmentation done later, but only for right arms, for left arms 
    ## it can get worse. 
    ## the problem usually occurs for the "thumb" gesture (I think it is a problem for the
    ## cases that do not have many forearm points, but (because of the gesture) are above
    ## the threshold ratio, finding a better way than the ratio
    ## to identify when to not do the segmentation should be a good improvement
    c2 = -np.abs(x[1, 0])
    x = np.array([c1, c2])
    ## possible improvement by creating a mask
    hand_mask = np.ones(np.array(hand.shape))
    for i in range(hand_mask.shape[0]):
        for j in range(hand_mask.shape[1]):
            # transform the coordinates of the points from the 'image space'
            # to the 'transformed space' using the transformation matrix 
            # if the point is greater than the segmentation point it is discarded
            if np.dot(np.array([i, j]) - absolute_offset, x) > y[segIndex, 0]:
                hand_mask[i, j] = 0

    return hand_mask


def hd_resample_points(sample, N=1024):
    work_sample = deepcopy(sample)
    points_to_add = []
    non_zero = work_sample > 0
    number_of_points = non_zero.sum()
    while number_of_points != N:
        # kernel
        x, y = np.where(non_zero)
        neighbors = np.zeros(np.array(non_zero.shape))
        for i in range(x.size):
            kernel_size = 1
            while True:
                if ((x[i] - kernel_size) >= 0) and ((x[i] + kernel_size) < non_zero.shape[0]):
                    if ((y[i] - kernel_size) >= 0) and ((y[i] + kernel_size) < non_zero.shape[1]):
                        window = non_zero[x[i] - kernel_size: x[i] + kernel_size + 1,
                                 y[i] - kernel_size: y[i] + kernel_size + 1]
                        if window.sum() < (kernel_size * 2 + 1) ** 2:
                            break
                    else:
                        window = [1]
                        break
                else:
                    window = [1]
                    break

                kernel_size = kernel_size + 1

            neighbors[x[i], y[i]] = np.array(window).sum() - 1

        flags = []
        if number_of_points > N:
            flags = neighbors == neighbors.max()

            flagsx, flagsy = np.where(flags)
            index = int(np.random.random() * flagsx.shape[0])
            coord = [flagsx[index], flagsy[index]]

            work_sample[coord[0], coord[1]] = 0
            number_of_points = number_of_points - 1
            non_zero[coord[0], coord[1]] = False
        else:
            neighbors_data = neighbors.reshape([neighbors.size])
            neighbors_data = neighbors_data[neighbors_data != 0]
            a, b = np.histogram(neighbors_data, 100)
            hist_res = b[1] - b[0]
            # in python b contains the edges of the bars, while in matlab
            # b contains the center, to account for this, we add half the
            # thickness to the left edge to obtain the center, and we discard
            # the last right edge
            b = b[:-1] + hist_res / 2
            c, = np.where(np.cumsum(a) > 0.5 * a.sum())
            threshold = np.round(b[c[0]])
            flags = (neighbors >= threshold - np.ceil(hist_res / 2)) * (neighbors <= threshold + np.ceil(hist_res / 2))

            flagsx, flagsy = np.where(flags)
            index = int(np.random.random() * flags.shape[0])
            coord = [flagsx[index], flagsy[index]]

            points_to_add.append([coord[0], coord[1], work_sample[coord[0], coord[1]]])
            number_of_points = number_of_points + 1

    points_list = hd_take_out_zeros(work_sample)
    if len(points_to_add) > 0:
        points_list = np.append(points_list, np.array(points_to_add), axis=0)

    return points_list


def hd_resample_points2(sample, N=1024):
    work_sample = deepcopy(sample)
    points_to_add = []
    non_zero = work_sample > 0
    number_of_points = non_zero.sum()
    if number_of_points != N:
        # kernel
        x, y = np.where(non_zero)
        graph_shape = list(non_zero.shape)
        graph_shape.extend(non_zero.shape)
        graph_shape = np.array(graph_shape)
        # repeating the same while loop for every point added or removed takes
        # a lot of time, instead we can memorise for each point its neighbors
        # to do this in a easy way we can treat it like a graph and use the
        # matrix representation.
        # since we work with points in an image (that have 2 coordinates: x and y)
        # we will have a 4 dimensional matrix.
        # the first 2 dimensions are the coordinates of the point we ar working on
        # the and the next 2 dimensions are the coordinates of its neighbors
        # we will store a 1 if they are neighbors and 0 otherwise
        # example: point A(x=23,j=36) is neighbor B(x=26,j=40)
        # neighbors(23,36,26,40) is going to be 1
        neighbors = np.zeros(graph_shape)
        neighbors_number = np.zeros(np.array(non_zero.shape))
        for i in range(x.size):
            kernel_size = 1
            while True:
                if ((x[i] - kernel_size) >= 0) and ((x[i] + kernel_size) < non_zero.shape[0]):
                    if ((y[i] - kernel_size) >= 0) and ((y[i] + kernel_size) < non_zero.shape[1]):
                        # here we extract the window (size of the kernel) to check if it is full
                        # of neighbors
                        window = non_zero[x[i] - kernel_size: x[i] + kernel_size + 1,
                                 y[i] - kernel_size: y[i] + kernel_size + 1]
                        if window.sum() < (kernel_size * 2 + 1) ** 2:
                            break
                    else:
                        window = np.array([1])
                        break
                else:
                    window = np.array([1])
                    break

                kernel_size = kernel_size + 1

            # here we save our window in our representation matrix
            kernelSize = int((window.shape[0] - 1) / 2)
            neighbors[x[i], y[i], x[i] - kernelSize: x[i] + kernelSize + 1,
            y[i] - kernelSize: y[i] + kernelSize + 1] = window
            neighbors_number[x[i], y[i]] = window.sum() - 1

        while number_of_points > N:
            flags = neighbors_number == neighbors_number.max()

            flagsx, flagsy = np.where(flags)
            index = int(np.random.random() * flagsx.shape[0])
            coord = [flagsx[index], flagsy[index]]

            work_sample[coord[0], coord[1]] = 0

            number_of_points = number_of_points - 1

            # removing the point
            neighbors_number[coord[0], coord[1]] = 0
            # we remove all the neighbors of the point from the matrix
            neighbors[coord[0], coord[1], :, :] = 0
            # we remove the point from every points neighbor list
            # neighbors[:, :, coord[0], coord[1]] = 0

            # we need the points for which to recalculate the number of neighbors
            # since removing the point, doesn't just decrease the numbers of neighbors
            # by 1 but also limits the size of the kernel that calculates the number
            # which must be taken in consideration
            points_to_update = np.where(neighbors[:, :, coord[0], coord[1]] == 1)
            points_to_update = np.array(points_to_update)
            # for each point that must be updated
            for x, y in zip(points_to_update[0], points_to_update[1]):
                # we need to know the distance to the deleted point in order to
                # know the size of the kernel
                dist_to_point = np.max([abs(coord[0] - x), abs(coord[1] - y)])
                # we create a mask of the kernel for the neighbors we are keeping
                # every neighbor outside this mask will be removed
                mask = np.zeros(np.array(non_zero.shape))
                mask[x - dist_to_point: x + dist_to_point + 1, y - dist_to_point: y + dist_to_point + 1] = 1
                neighbors[x, y, :, :] = neighbors[x, y, :, :] * mask
                # recalculating the number of neighbors
                neighbors_number[x, y] = neighbors[x, y, :, :].sum() - 1

        while number_of_points < N:
            neighbors_data = neighbors_number.reshape([neighbors_number.size])
            neighbors_data = neighbors_data[neighbors_data != 0]
            a, b = np.histogram(neighbors_data, 100)
            hist_res = b[1] - b[0]
            # in python b contains the edges of the bars, while in matlab
            # b contains the center, to account for this, we add half the
            # thickness to the left edge to optain the center, and we discard
            # the last right edge
            b = b[:-1] + hist_res / 2
            c, = np.where(np.cumsum(a) > 0.5 * a.sum())
            threshold = np.round(b[c[0]])
            flags = (neighbors_number >= threshold - np.ceil(hist_res / 2)) * (
                    neighbors_number <= threshold + np.ceil(hist_res / 2))

            flagsx, flagsy = np.where(flags)
            index = int(np.random.random() * flagsx.shape[0])
            coord = [flagsx[index], flagsy[index]]

            points_to_add.append([coord[0], coord[1], work_sample[coord[0], coord[1]]])
            number_of_points = number_of_points + 1

            # updateing the number of neighbors
            # neighbors_number[coord[0], coord[1]] = neighbors_number[coord[0], coord[1]] + 1
            points_to_update = np.where(neighbors[:, :, coord[0], coord[1]] == 1)
            # for each point that must be updated
            for x, y in zip(points_to_update[0], points_to_update[1]):
                neighbors[x, y, coord[0], coord[1]] = neighbors[x, y, coord[0], coord[1]] + 1
                # recalculating the number of neighbors
                neighbors_number[x, y] = neighbors[x, y, :, :].sum() - 1

    points_list = hd_take_out_zeros(work_sample)
    if len(points_to_add) > 0:
        points_list = np.append(points_list, np.array(points_to_add), axis=0)

    return points_list


def hd_scale(point_list, hw_ratio=0.2):
    x = np.array(point_list[:, 0])
    y = np.array(point_list[:, 1])
    z = np.array(point_list[:, 2])

    xShifted = x - (x.max() + x.min()) / 2
    yShifted = y - (y.max() + y.min()) / 2
    zShifted = z - (z.max() + z.min()) / 2

    xyScale = np.max(np.abs([xShifted, yShifted]))
    zScale = np.max(np.abs(zShifted))

    xScaled = xShifted / xyScale
    yScaled = yShifted / xyScale
    zScaled = zShifted / zScale * hw_ratio

    return np.array([xScaled, yScaled, zScaled]).T


# the image must be monochrome
def run_on_sample(dist_data, image_data=None):
    ## I intended to make the function to return a mask
    ## to be used to extract the hand from a paired image 
    ## (regardles if it is monochrome, colored or anything else)
    ## but the curent implementation of some functions makes this
    ## difficult and inefficient to the point where 
    ## processing the image in parallel is a better option
    ## doing this would make the program more modular
    ## and easier to use

    image_present = False
    if image_data is not None:
        image_present = True

    f_sample = hd_filter(dist_data)

    # avg_dist = f_sample.mean()
    # foreground_mask = f_sample < avg_dist
    #f_sample_person = f_sample * foreground_mask

    f_sample_person_box = hd_rectangle(f_sample)
    h_sample, flag = hd_getClosestPoint(f_sample_person_box)
    if CLI_VERBOSE:
        print("Number of skipped points: %d" % (flag))

    if len(h_sample) == 0:
        return [], [], False

    r_sample = hd_rectangle(h_sample)

    filtered_regions = hd_extract_regions_of_interest(r_sample)

    if image_present:
        image_data = hd_rectangle(image_data * foreground_mask)
        image_data = hd_rectangle(image_data * (h_sample > 0))

    if len(filtered_regions) > 0:
        if len(filtered_regions) > 1:
            pass
        for region in filtered_regions:

            minr, minc, maxr, maxc = region.bbox
            try:
                hand_mask = hd_remove_forearm(region.image)

                hand = hd_rectangle(hand_mask * r_sample[minr:maxr, minc:maxc])

                if image_present:
                    image_data = hd_rectangle(image_data[minr:maxr, minc:maxc] * hand_mask)

                points_list = hd_resample_points2(hand)
                scaled_points_list = hd_scale(points_list)

                if CLI_VERBOSE:
                    print("Number of points: %d" % ((hand > 0).sum()))

                if SHOW_IMAGES:
                    plotDist(dist_data, False)
                    plotCentroids(r_sample, filtered_regions, False)
                    show_points_list = scaled_points_list
                    show_points_list[:, 2] = - show_points_list[:, 2]
                    plot3D(show_points_list, False)
                    if image_present:
                        imshow(image_data)

            except Exception as e:
                return [], [], False

        if SHOW_IMAGES:
            showPlots()

        return scaled_points_list, image_data, True

    else:
        return [], [], False


def main():
    samples = load_data()
    processed_samples = []
    PARALLEL = False
    NUMBER_OF_THREADS = 4

    if PARALLEL:
        SHOW_IMAGES = False
        CLI_VERBOSE = False
    tic = time()

    if not PARALLEL:
        for i, sample in enumerate(samples):
            print("sample: %d - gesture %d" % (i, sample[2]))
            out_hand, out_img, success = run_on_sample(sample[0], sample[1])
            if success:
                processed_samples.append([out_hand, out_img, sample[2]])
            if i == 10:
                break
    else:
        queues = []
        threaded_run_on_sample = threaded(run_on_sample)
        for i, sample in enumerate(samples):
            print("sample: %d - gesture %d" % (i, sample[2]))
            queues.append((threaded_run_on_sample(sample[0], sample[1]), sample[2]))
            if i % NUMBER_OF_THREADS == NUMBER_OF_THREADS - 1:
                for q in queues:
                    out_hand, out_img, success = q[0].get()
                    if success:
                        processed_samples.append([out_hand, out_img, q[1]])
                queues = []
        for q in queues:
            out_hand, out_img, success = q[0].get()
            if success:
                processed_samples.append([out_hand, out_img, q[1]])
        queues = []

    toc = time() - tic
    print(toc)
    __ = input('')
    data = []
    images = []
    labels = []
    for i, sample in enumerate(processed_samples):
        data.append(sample[0])
        images.append(sample[1])
        labels.append(sample[2])

    data = np.array(data, 'float32')
    image = np.array(images, 'float32')
    labels = np.array(labels, 'uint8')
    save_h5("out.h5", data, labels)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    main()
    exit()
    tic = time()
    # dist_data=load_bin("data/02_dst.bin",[6,-1,200,200])
    # img_data=load_bin("data/02_int.bin",[6,-1,200,200])
    # pickle.dump(dist_data,open("dist_data_2.pickle","wb"))
    # pickle.dump(img_data,open("img_data_2.pickle","wb"))
    # dist_data.reshape([6,20,200,200])
    # dist_data=pickle.load(open("dist_data_2.pickle","rb"))
    # img_data=pickle.load(open("img_data_2.pickle","rb"))
    dist_data = pickle.load(open("dist_data.pickle", "rb"))
    img_data = pickle.load(open("img_data.pickle", "rb"))
    #
    # # showImage(images[0]).show()

    # # input("")
    # data_sample=pickle.load(open("sample5.pickle","rb"))
    # sample=(data_sample["dist_data"],data_sample["img_data"])
    imgs = []
    fails = []
    for i in range(3, dist_data.shape[0]):
        for j in range(0, dist_data.shape[1]):
            try:
                # 5 0
                # 2 18
                # 2 19
                # 3 16 - 3 19
                # i=3
                # j=16
                # i=3;j=5
                # i=1;#j=1
                sample = (dist_data[i, j], img_data[i, j], i, j)
                __, img = run_on_sample(sample[0], sample[1])
                # imgs.append(run_on_sample(sample))
                imgs.append((i, j, img))
                print((i, j))
            except Exception as e:
                print(e)
                fails.append((i, j))
            # a=input()
            # if 'e' in a:
            #     quit()
    # imgs2=[]
    # for img in imgs:
    #     imgs2.append(img.get())

    toc = time() - tic
    # pickle.dump(imgs,open("output.pickle","wb"))
    print(toc)
    input()
