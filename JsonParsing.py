import json
import os
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
    return Image(json_dict["image_name"], json_dict["c1_feature"], json_dict["c2_feature"], json_dict["c3_feature"],
                 json_dict["distance"])


def save_to_json(filename, image_base, time_taken):
    print("Writing image base features to:", filename)
    file = open(filename, 'w')
    file.write(time_taken + "\n")
    for image in image_base:
        file.write(json.dumps(image, cls=ImageEncoder) + "\n")
    file.close()
    print("Finished writing image features")


# TODO: Handel errors
def load_for_json(filename):
    image_base = []
    print("Opening file:", filename)
    # file automatically closed
    with open(filename, 'r') as file:
        # time take to encode features first entry in file
        time_taken = file.readline()
        print("Reading in images")
        for line in file:
            image_base.append(json.loads(line, object_hook=image_decoder))
        print("Finished loading image features")
    return image_base


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
    print("start")
    image_base = load_for_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/100.json")
    print("finish")
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


if __name__ == '__main__':
    main()
