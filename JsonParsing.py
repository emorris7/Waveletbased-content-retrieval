import json
import os
import time
from json import JSONEncoder

import numpy as np

from ProcessImage import Image


# https://pynative.com/python-convert-json-data-into-custom-python-object/
class ImageEncoder(JSONEncoder):
    def default(self, o):
        # for encoding coefficients
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o.__dict__


def image_decoder(json_dict):
    image = Image(image_name=json_dict["image_name"], c1_feature=json_dict["c1_feature"],
                  c2_feature=json_dict["c2_feature"], c3_feature=json_dict["c3_feature"],
                  distance=json_dict["distance"])
    return image


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


# TODO: Handel errors
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


def make_json_4_2(folder, level, long):
    image_base = []
    # TODO: error if not image
    print("Loading image base...")
    count = 0
    # Timer for creating database of features
    for filename in os.listdir(folder):
        # computing level 4 dwt
        Image(image_name=filename, level=level, long=long)
        count += 1
        print(count)
    start_time = time.perf_counter()
    for filename in os.listdir(folder):
        # computing level 4 dwt
        image_base.append(Image(image_name=filename, level=level, long=long))
        count += 1
        print(count)
    end_time = time.perf_counter()
    loading_time = end_time - start_time
    print("Finished loading images. Time taken:", loading_time)
    save_to_json("testy.json", image_base, str(loading_time), "no_long 4 db2 periodization")


def main():
    # image1 = Image("000007.jpg")
    # image2 = Image("000009.jpg")
    # image3 = Image("000012.jpg")
    # image_base_eg = [image1, image2, image3]
    # save_to_json("test.json", image_base_eg, "3.4")
    # image_base_test = load_for_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/test.json")
    # image1.image_distance(image_base_test[0], [1, 1, 1, 1], 1, 1, 1)
    # image2.image_distance(image_base_test[1], [1, 1, 1, 1], 1, 1, 1)
    # image3.image_distance(image_base_test[2], [1, 1, 1, 1], 1, 1, 1)
    # print(image_base_test[0].image_name, image_base_test[1].image_name, image_base_test[2].image_name)
    # print(image_base_test[0].c1_feature[0], image_base_test[0].c2_feature[0], image_base_test[0].c3_feature[0])
    # print(image1.c1_feature[0], image1.c2_feature[0], image1.c3_feature[0])
    # print(image_base_test[1].c1_feature[0], image_base_test[1].c2_feature[0], image_base_test[1].c3_feature[0])
    # print(image2.c1_feature[0], image2.c2_feature[0], image2.c3_feature[0])
    # print(image_base_test[2].c1_feature[0], image_base_test[2].c2_feature[0], image_base_test[2].c3_feature[0])
    # print(image3.c1_feature[0], image3.c2_feature[0], image3.c3_feature[0])
    # print(image1.distance, image2.distance, image3.distance)
    # print("start")
    # image_base = load_for_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/100.json")
    # print("finish")
    # folder = "/home/emily/Documents/2021/CSC5029Z/MiniProject/100"
    # # TODO: error if not image
    # print("Loading image base...")
    # # count = 0
    # for filename in os.listdir(folder):
    #     image_base.append(Image(filename))
    #     # count += 1
    #     # print(count)
    # print("Finished loading images")
    # print(len(image_base))
    # save_to_json("100.json", image_base, "3.4")
    # print("finished saving")
    # LOADING CORAL1-K db4 two check periodization

    make_json_4_2("/home/emily/Documents/2021/CSC5029Z/MiniProject/100", level=4, long=False)


if __name__ == '__main__':
    main()
