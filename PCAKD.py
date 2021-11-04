import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

import JsonParsing
import ProcessImage


# Find the best num_matches images matching the specified by query name. Results returned are a list of PCAImages
# This is mainly just used for testing and comparison to the improve KD-tree implementation
def find_images_pca(stand_scaler, pca_images, image_base, query_image_name, num_matches, level=4, long=False):
    # start timers
    start_time = time.perf_counter()
    # create a single feature vector for the image
    query_image_feature = create_single_feature(query_image_name, level, long)
    # standardize the feature vector
    standardised_feature = stand_scaler.transform([query_image_feature])
    # project feature vector onto the new basis
    reduced_feature = pca_images.transform(standardised_feature)
    query_image = ProcessImage.PCAImage(image_name=query_image_name, feature=reduced_feature[0])
    # compute distance
    for img in image_base:
        img.image_distance(query_image)
    # order based on distance
    image_base.sort(key=lambda img: img.distance)
    stop_time = time.perf_counter()
    compute_time = stop_time - start_time
    return image_base[:num_matches:], compute_time


# Find the best num_matches images matching the specified by query name. Results returned are a list of image names
def find_images_kd(stand_scaler, pca_images, kd_tree, image_names, query_image_name, num_matches, level=4, long=False):
    # start timers
    start_time = time.perf_counter()
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
    compute_time = stop_time - start_time
    return found_images, compute_time


# Create an image and flatten its features to create a single 1d feature
def create_single_feature(filename, level=4, long=False):
    image = ProcessImage.Image(image_name=filename, level=level, long=long)
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


# Flatten the features of the image to created a single 1d image features
def create_single_feature_image(image):
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


# Extract all the 1d features and image names from a directory of images
def extract_features(folder, level=4, long=False):
    data_array = []
    image_names = []
    count = 0
    for filename in os.listdir(folder):
        data_array.append(create_single_feature(filename, level, long))
        image_names.append(filename)
        count += 1
        print(count)
    return data_array, image_names


# Construct the reduced database using PCA
def pca_database(data_array, explained=0.2):
    # standardize the features https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    stand_scaler = StandardScaler().fit(data_array)
    standardised_data = stand_scaler.transform(data_array)
    pca_images = PCA(n_components=explained).fit(standardised_data)
    reduced_data = pca_images.transform(standardised_data)
    return stand_scaler, pca_images, reduced_data


def main():
    stand_scaler, pca_images, image_names, tree = [], [], [], []
    if len(sys.argv) <= 1:
        print("Loading image base features from json file")
        stand_scaler, pca_images, image_names, tree, time_taken = JsonParsing.load_kd_pca(
            "Json_Database/Corel1-k_db2_period_no_long.json")
        print("Feature base created. Time taken to create original feature base: ", time_taken)
    else:
        folder = sys.argv[1]
        start_time = time.perf_counter()
        data_array, image_names = extract_features(folder)
        stand_scaler, pca_images, reduced_data = pca_database(data_array)
        # https: // scikit - learn.org / stable / modules / generated / sklearn.neighbors.KDTree.html
        tree = KDTree(reduced_data, leaf_size=2)
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        print("Finished loading images. Time taken:", loading_time)
    while True:
        option = input("Enter a query image (or 'quit' to exit): ")
        if option == "quit":
            break
        else:
            # TODO: change beta threshold based on number of matches
            matches = input("Enter number of matches to search for: ")
            if matches.isdigit():
                matches = int(matches)
                if matches > len(image_names):
                    print("Not enough images. Number of requested matches:", matches, "Number of images to search: ",
                          len(image_names))
                else:
                    matching_images, time_taken = find_images_kd(stand_scaler=stand_scaler,
                                                                 pca_images=pca_images,
                                                                 kd_tree=tree,
                                                                 image_names=image_names,
                                                                 query_image_name=option,
                                                                 num_matches=matches)
                    print("Best matches found. Time taken:", time_taken)
                    print("Precision:", calc_rp_kd(option, matching_images, matches))
                    view_image = input("View the best matching images (y\\n): ")
                    if view_image.lower() == "y":
                        ProcessImage.show_images(option, matching_images, True)
                    save_image = input("Save the best matching images (y\\n): ")
                    if save_image.lower() == "y":
                        save_image_name = input("Enter name for image to be saved as: ")
                        ProcessImage.save_images(option, matching_images, save_image_name + ".png", True)
                        print("Images saved to", save_image_name + ".png")
            else:
                print("Invalid match number specified:", matches, ".Enter an integer")


# Calculate the average precision for the returned images
# Assuming the images are divided into semantic classes and that the class is denoted by the first letter/number of the
# image name
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
