import json
import os
import time
from json import JSONEncoder

import numpy as np
from sklearn.neighbors import KDTree

import PCAKD
import ProcessImage


# https://pynative.com/python-convert-json-data-into-custom-python-object/
class ImageEncoder(JSONEncoder):
    def default(self, o):
        # for encoding coefficients
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o.__dict__


def image_decoder(json_dict):
    image = ProcessImage.Image(image_name=json_dict["image_name"], c1_feature=json_dict["c1_feature"],
                               c2_feature=json_dict["c2_feature"], c3_feature=json_dict["c3_feature"],
                               distance=json_dict["distance"])
    return image


# Saves Images represented by features to a json file
def save_to_json(filename, image_base, time_taken, long_level_string):
    print("Writing image base features to:", filename)
    file = open(filename, 'w')
    file.write(time_taken + "\n")
    # denotes what level and what depth is used
    file.write(long_level_string + "\n")
    for image in image_base:
        file.write(json.dumps(image, cls=ImageEncoder) + "\n")
    file.close()
    print("Finished writing image features")


# Load image features and build an image base from a json file
def load_from_json(filename):
    image_base = []
    print("Opening file:", filename)
    # file automatically closed
    with open(filename, 'r') as file:
        # time take to encode features first entry in file
        time_taken = file.readline()
        # level of decomposition used and whether adding composition was used encoded in a string, wavelet used, mode of extension
        parameters_string = file.readline()
        print("Reading in images")
        for line in file:
            image_base.append(json.loads(line, object_hook=image_decoder))
        print("Finished loading image features")
    return image_base


# Load image features, perform PCA and produce a KD_tree for the images
def load_kd_pca(filename):
    image_base = load_from_json(filename)
    features = [PCAKD.create_single_feature_image(f) for f in image_base]
    image_names = [img.image_name for img in image_base]
    stand_scaler, pca_images, reduced_data = PCAKD.pca_database(features)
    # https: // scikit - learn.org / stable / modules / generated / sklearn.neighbors.KDTree.html
    tree = KDTree(reduced_data, leaf_size=2)
    return stand_scaler, pca_images, image_names, tree


def make_json_4_2(folder, level, long):
    image_base = []
    # TODO: error if not image
    print("Loading image base...")
    count = 0
    # Timer for creating database of features
    for filename in os.listdir(folder):
        # computing level 4 dwt
        ProcessImage.Image(image_name=filename, level=level, long=long)
        count += 1
        print(count)
    start_time = time.perf_counter()
    for filename in os.listdir(folder):
        # computing level 4 dwt
        image_base.append(ProcessImage.Image(image_name=filename, level=level, long=long))
        count += 1
        print(count)
    end_time = time.perf_counter()
    loading_time = end_time - start_time
    print("Finished loading images. Time taken:", loading_time)
    save_to_json(folder + ".json", image_base, str(loading_time), "no_long 4 db2 periodization")


def main():
    make_json_4_2("/home/emily/Documents/2021/CSC5029Z/MiniProject/VOC_Subset", level=4, long=False)


if __name__ == '__main__':
    main()
