import cv2
import numpy as np
import pywt


class Image:

    # Take in a variable number of parameters to allow for loading features from json
    # long determines whether an extra level of dwt is computed or not
    # level determines the minimum level of dwt that will be done
    # Acceptable inputs are:
    # [image_name, level, long] or
    # [image_name, c1_feature, c2_feature, c3_feature, distance]
    def __init__(self, *args):
        if len(args) == 3:
            self.image_name = args[0]
            level = args[1]
            long = args[2]
            # TODO: CHANGE
            c1, c2, c3 = self.create_axes("/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral1-k/" + args[0])
            # Feature is array [standard_deviation, dA5, dH5, dV5, dD5, dA4, dH4, dV4, dD4]
            self.c1_feature = self.create_features_long(c1, level) if long else self.create_features(c1, level)
            self.c2_feature = self.create_features_long(c2, level) if long else self.create_features(c2, level)
            self.c3_feature = self.create_features_long(c3, level) if long else self.create_features(c3, level)
            self.distance = 0
        else:
            self.image_name = args[0]
            self.c1_feature = args[1]
            self.c2_feature = args[2]
            self.c3_feature = args[3]
            self.distance = args[4]

    # Takes in an image name, resizes image and extracts 3 color axes using the formulas:
    # C1 = (R+G+B)/3
    # C2 = (R+(max-B))/2
    # C3 = (R +2*(max-G)+b)/4
    # Returns 3 matrices, describing each of the color axes
    # TODO: potentially change to some other axis
    def create_axes(self, image_name):
        original = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        # resize image, using interlinear interpolation
        # TODO: potentially change size and interpolation
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

    # Extract features from the color axis of an image as defined in:
    # "Content-based image indexing and searching using Daubechies' wavelets"
    # Features:
    # 1. 16x16 sub-matrix of 4-level DWT: M4
    # 2. Standard deviation of 8x8 sub-matrix of M4
    # 3. 8x8 sub-matrix of 5-level DWT: M5
    def create_features_long(self, color_axis, level):
        # leave mode as symmetric, use debauchie 2 wavelets
        # TODO: Potentially change wavelet type and signal extension method
        dwt4 = pywt.wavedec2(color_axis, 'db2', level=level)
        # approximation and detail matrices for level 4
        dA4 = dwt4[0]
        (dH4, dV4, dD4) = dwt4[1]
        standard_deviation = dA4.std()
        dwt5 = pywt.wavedec2(color_axis, 'db2', level=(level + 1))
        # approximation and detail for level 5
        dA5 = dwt5[0]
        (dH5, dV5, dD5) = dwt5[1]
        return standard_deviation, dA5, dH5, dV5, dD5, dA4, dH4, dV4, dD4

    def create_features(self, color_axis, level):
        # leave mode as symmetric, use debauchie 2 wavelets
        # TODO: Potentially change wavelet type and signal extension method
        # dwt4 = pywt.wavedec2(color_axis, 'db2', mode='periodization', level=4)
        dwt4 = pywt.wavedec2(color_axis, 'db2', level=level)
        # approximation and detail matrices for level 4
        dA4 = dwt4[0]
        (dH4, dV4, dD4) = dwt4[1]
        standard_deviation = dA4.std()
        return standard_deviation, dA4, dH4, dV4, dD4

    def filter_standard_deviation(self, query_image, percent=50):
        beta = 1 - percent / 100
        # filter based on standard deviation
        c1_std, c2_std, c3_std = self.c1_feature[0], self.c2_feature[0], self.c3_feature[0]
        c1q_std, c2q_std, c3q_std = query_image.c1_feature[0], query_image.c2_feature[0], query_image.c3_feature[0]
        if ((c1_std < (beta * c1q_std) or c1_std > (c1q_std / beta)) and (
                c2_std < (beta * c2q_std) or c2_std > (c2q_std / beta)) or c3_std < (
                beta * c3q_std) or c3_std > (c3q_std / beta)):
            # no longer consider image
            # print(c1_std, c2_std, c3_std)
            # print(c1q_std, c2q_std, c3q_std)
            self.distance = -1
        else:
            self.distance = 0

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
