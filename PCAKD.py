import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from ProcessImage import Image
from ProcessImage import PCAImage

# Results returned are PCAImages
def find_images(stand_scaler, pca_images, image_base, query_image_name, num_matches, level, long):
    # start timers
    start_time = time.perf_counter()
    start_time_p = time.process_time()
    # create a single feature vector for the image
    query_image_feature = create_single_feature(query_image_name, level, long)
    # standardize the feature vector
    standardised_feature = stand_scaler.transform([query_image_feature])
    # project feature vector onto the new basis
    reduced_feature = pca_images.transform(standardised_feature)
    query_image = PCAImage(image_name=query_image_name, feature=reduced_feature[0])
    # compute distance
    for img in image_base:
        img.image_distance(query_image)
    # order based on distance
    image_base.sort(key=lambda img: img.distance)
    stop_time = time.perf_counter()
    stop_time_p = time.process_time()
    compute_time = stop_time - start_time
    compute_time_p = stop_time_p - start_time_p
    return image_base[:num_matches:], compute_time, compute_time_p

# Results returned are a list of image names
def find_images_kd(stand_scaler, pca_images, kd_tree, image_names, query_image_name, num_matches, level, long):
    # start timers
    start_time = time.perf_counter()
    start_time_p = time.process_time()
    # create a single feature vector for the image
    query_image_feature = create_single_feature(query_image_name, level, long)
    # standardize the feature vector
    standardised_feature = stand_scaler.transform([query_image_feature])
    # project feature vector onto the new basis
    reduced_feature = pca_images.transform(standardised_feature)
    dist, indexes = kd_tree.query(reduced_feature, k=num_matches)
    # get the names of the images that are matched
    found_images = [image_names[x] for x in indexes[0]]
    stop_time = time.perf_counter()
    stop_time_p = time.process_time()
    compute_time = stop_time - start_time
    compute_time_p = stop_time_p - start_time_p
    return found_images, compute_time, compute_time_p


def create_single_feature(filename, level, long):
    image = Image(image_name=filename, level=level, long=long)
    c1_feature, c2_feature, c3_feature = image.c1_feature, image.c2_feature, image.c3_feature
    # feature: standard_deviation, dA4, dH4, dV4, dD4 || standard_deviation, dA5, dH5, dV5, dD5, dA4, dH4, dV4, dD4
    c1_vector = np.concatenate(list(
        map(lambda x: np.concatenate(x) if (isinstance(x, np.ndarray) or isinstance(x, list)) else [x],
            c1_feature)))
    c2_vector = np.concatenate(list(
        map(lambda x: np.concatenate(x) if (isinstance(x, np.ndarray) or isinstance(x, list)) else [x],
            c2_feature)))
    c3_vector = np.concatenate(list(
        map(lambda x: np.concatenate(x) if (isinstance(x, np.ndarray) or isinstance(x, list)) else [x],
            c3_feature)))
    return np.concatenate([c1_vector, c2_vector, c3_vector])


def extract_features(folder, level, long):
    data_array = []
    image_names = []
    count = 0
    for filename in os.listdir(folder):
        data_array.append(create_single_feature(filename, level, long))
        image_names.append(filename)
        count += 1
        if (count % 50) == 0:
            print(count)
    return data_array, image_names


def pca_database(data_array, explained=0.2):
    # standardize the features https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    stand_scaler = StandardScaler().fit(data_array)
    standardised_data = stand_scaler.transform(data_array)
    pca_images = PCA(n_components=explained).fit(standardised_data)
    reduced_data = pca_images.transform(standardised_data)
    # print(reduced_data)
    return stand_scaler, pca_images, reduced_data


def main():
    folder_load = "/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral1-k"
    folder_search = "/home/emily/Documents/2021/CSC5029Z/MiniProject/subsets"
    level = 4
    long = False
    data_array, image_names = extract_features(folder_load, level, long)
    stand_scaler, pca_images, reduced_data = pca_database(data_array)
    # https: // scikit - learn.org / stable / modules / generated / sklearn.neighbors.KDTree.html
    tree = KDTree(reduced_data, leaf_size=2)
    while True:
        option = input("Enter a query image (or 'quit' to exit): ")
        if option == "quit":
            break
        else:
            # TODO: change beta threshold based on number of matches
            matches = eval(input("Enter number of matches to search for: "))
            if matches > len(image_names):
                print("Not enough images. Number of requested matches:", matches, "Number of images to search: ",
                      len(image_names))
            else:
                # print("Beginning image retrieval")
                matching_images, time_taken, time_taken_p = find_images_kd(stand_scaler=stand_scaler,
                                                                           pca_images=pca_images,
                                                                           kd_tree=tree, image_names=image_names,
                                                                           query_image_name=option, num_matches=matches,
                                                                           level=level,
                                                                           long=long)
                print("Time taken:", time_taken, time_taken_p)
                # print("Best matching images")
                print(calc_rp_kd(option, matching_images, matches))
                # for img in matching_images:
                #     print(img.image_name, img.distance)


def calc_rp_kd(query_name, returned_results, num_matches):
    # use start number as class identifier
    start_letter = query_name[0]
    num_correct = 0
    for image in returned_results:
        if image[0] == start_letter:
            num_correct += 1
    return num_correct / num_matches


if __name__ == '__main__':
    main()
