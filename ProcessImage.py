import itertools

import cv2
import numpy as np
import pywt


class Image:
    # Take in a variable number of parameters to allow for loading features from json
    # long determines whether an extra level of dwt is computed or not
    # level determines the minimum level of dwt that will be done
    # Acceptable inputs are:
    # [image_name] or
    # [image_name, c1_feature, c2_feature, c3_feature, distance]
    def __init__(self, image_name=None, level=4, long=False, mode="periodization", wave="db2", c1_feature=None,
                 c2_feature=None,
                 c3_feature=None, distance=None):
        # TODO TEST CHANGE BACK TO 3
        if c1_feature is None:
            self.image_name = image_name
            # TODO: CHANGE
            c1, c2, c3 = self.create_axes("/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral1-k/" + image_name)
            # Feature is array [standard_deviation, dA5, dH5, dV5, dD5, dA4, dH4, dV4, dD4]
            self.c1_feature = self.create_features_long(c1, level, mode, wave) if long else self.create_features(c1,
                                                                                                                 level,
                                                                                                                 mode,
                                                                                                                 wave)
            self.c2_feature = self.create_features_long(c2, level, mode, wave) if long else self.create_features(c2,
                                                                                                                 level,
                                                                                                                 mode,
                                                                                                                 wave)
            self.c3_feature = self.create_features_long(c3, level, mode, wave) if long else self.create_features(c3,
                                                                                                                 level,
                                                                                                                 mode,
                                                                                                                 wave)
            self.distance = 0
        else:
            self.image_name = image_name
            self.c1_feature = c1_feature
            self.c2_feature = c2_feature
            self.c3_feature = c3_feature
            self.distance = distance

    # Takes in an image name, resizes image and extracts 3 color axes using the formulas:
    # C1 = (R+G+B)/3
    # C2 = (R+(max-B))/2
    # C3 = (R +2*(max-G)+b)/4
    # Returns 3 matrices, describing each of the color axes
    def create_axes(self, image_name):
        original = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        # resize image, using bilinear interpolation
        image = cv2.resize(original, (128, 128), interpolation=cv2.INTER_LINEAR)
        width, height, components = image.shape
        c1 = np.zeros((width, height))
        c2 = np.zeros((width, height))
        c3 = np.zeros((width, height))
        red = image[:, :, 2]
        green = image[:, :, 1]
        blue = image[:, :, 0]
        # calculate the axes, values will be real numbers (not necessarily whole numbers)
        for i in range(width):
            for j in range(height):
                c1[i][j] = (int(red[i][j]) + int(green[i][j]) + int(blue[i][j])) / 3
                c2[i][j] = (red[i][j] + (255 - blue[i][j])) / 2
                c3[i][j] = (red[i][j] + 2 * (255 - green[i][j]) + blue[i][j]) / 4
        return c1, c2, c3

    # Extract features from the color axis of an image using both a base level and an extra level of decomposition for DWT
    # Features:
    # 1. Sub-matrix of level-(level) DWT: M4
    # 2. Standard deviation of sub-matrix of M4
    # 3. Sub-matrix of level-(level+1) DWT: M5
    def create_features_long(self, color_axis, level, mode, wavelet):
        # leave mode as symmetric, use debauchie 2 wavelets
        dwt4 = pywt.wavedec2(color_axis, wavelet=wavelet, mode=mode, level=level)
        # approximation and detail matrices for level
        dA4 = dwt4[0]
        (dH4, dV4, dD4) = dwt4[1]
        standard_deviation = dA4.std()
        dwt5 = pywt.wavedec2(color_axis, wavelet=wavelet, mode=mode, level=(level + 1))
        # approximation and detail for level + 1
        dA5 = dwt5[0]
        (dH5, dV5, dD5) = dwt5[1]
        return standard_deviation, dA5, dH5, dV5, dD5, dA4, dH4, dV4, dD4

    # Extract features from the color axis of an image using a specified level of DWT
    # Features:
    # 1. Sub-matrix of level-(level) DWT: M4
    # 2. Standard deviation of sub-matrix of M4
    def create_features(self, color_axis, level, mode, wavelet):
        # leave mode as symmetric, use debauchie 2 wavelets
        # TODO: Potentially change wavelet type and signal extension method
        # dwt4 = pywt.wavedec2(color_axis, 'db2', mode='periodization', level=4)
        dwt4 = pywt.wavedec2(color_axis, wavelet=wavelet, mode=mode, level=level)
        # approximation and detail matrices for level 4
        dA4 = dwt4[0]
        (dH4, dV4, dD4) = dwt4[1]
        standard_deviation = dA4.std()
        return standard_deviation, dA4, dH4, dV4, dD4

    # Compute whether image should be considered for further distance calculations based on whether the standard deviation values
    # for the colour axes are within certain ranges from the standard deviations of the query image
    def filter_standard_deviation(self, query_image, percent=50):
        beta = 1 - percent / 100
        # filter based on standard deviation
        c1_std, c2_std, c3_std = self.c1_feature[0], self.c2_feature[0], self.c3_feature[0]
        c1q_std, c2q_std, c3q_std = query_image.c1_feature[0], query_image.c2_feature[0], query_image.c3_feature[0]
        if ((c1_std < (beta * c1q_std) or c1_std > (c1q_std / beta)) and (
                c2_std < (beta * c2q_std) or c2_std > (c2q_std / beta)) or c3_std < (
                beta * c3q_std) or c3_std > (c3q_std / beta)):
            # no longer consider image
            self.distance = -1
        else:
            self.distance = 0

    # Calculate the distance from the query image using a weighted Euclidean distance
    # offset allows for either calculating difference for level i dwt or level i+1 dwt
    # offset = 1 (for i+1 dwt or i dwt if i+1 dwt not done)
    # offset = 5 (for i dwt if i+1 dwt done)
    # weights = [w11, w12, w21, w22]
    def image_distance(self, query_image, weights, wc1, wc2, wc3, offset=1):
        total_distance = 0
        for i in range(4):
            c1_diff = np.sum((np.subtract(query_image.c1_feature[i + offset], self.c1_feature[i + offset])) ** 2)
            c2_diff = np.sum((np.subtract(query_image.c2_feature[i + offset], self.c2_feature[i + offset])) ** 2)
            c3_diff = np.sum((np.subtract(query_image.c3_feature[i + offset], self.c3_feature[i + offset])) ** 2)
            sub_total = (wc1 * c1_diff) + (wc2 * c2_diff) + (wc3 * c3_diff)
            total_distance += (weights[i] * sub_total)
        self.distance = total_distance


# PCAImage used for storing information for images represented by reduced PCA features
class PCAImage:
    def __init__(self, image_name, feature):
        self.image_name = image_name
        self.feature = feature
        self.distance = 0

    def image_distance(self, query_image):
        self.distance = np.sum((np.subtract(self.feature, query_image.feature)) ** 2)


# Based on approach in https://gist.github.com/pgorczak/95230f53d3f140e4939c
# Put images in a grid to saving or displaying
def create_grid(image_names):
    num_images = len(image_names)
    # number of blank squares that need to be included to create a grid
    h = 11
    extra = num_images % h
    w = (num_images + (h - extra)) // h if extra != 0 else num_images // h
    imgs = [cv2.imread("/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral1-k/" + filename, cv2.IMREAD_UNCHANGED) for
            filename
            in image_names]
    resized = [cv2.resize(original, (128, 128), interpolation=cv2.INTER_LINEAR) for original in imgs]
    img_h, img_w, img_c = resized[0].shape

    # Pad images with blanks
    blank_matrix = np.zeros((img_h, img_w, img_c))
    if extra != 0:
        for i in range(h - extra):
            resized.append(blank_matrix)

    imgmatrix = np.zeros((img_w * w, img_h * h, img_c), np.uint8)

    imgmatrix.fill(255)

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, resized):
        x = x_i * (img_w)
        y = y_i * (img_h)
        imgmatrix[x:x + img_w, y:y + img_h, :] = img
    return imgmatrix


# View the query image and the resulting images in a grid with openCV with best matches appearing in the top row (ordered best
# to worst from left to right) and the quality of the matches decreasing as we descend down the rows
def show_images(query_image_name, returned_results, pca=False):
    image_names = []
    if pca:
        image_names = [query_image_name] + returned_results
    else:
        image_names = [query_image_name] + [x.image_name for x in returned_results]
    cv2.imshow("Best Matches", create_grid(image_names))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Save the query image and the resulting images in a grid with openCV with best matches appearing in the top row (ordered best
# to worst from left to right) and the quality of the matches decreasing as we descend down the rows
def save_images(query_image_name, returned_results, file_to_write, pca=False):
    image_names = []
    if pca:
        image_names = [query_image_name] + returned_results
    else:
        image_names = [query_image_name] + [x.image_name for x in returned_results]
    cv2.imwrite(file_to_write, create_grid(image_names))
