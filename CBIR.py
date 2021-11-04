import os
import sys
import time

import JsonParsing
from ProcessImage import Image, show_images, save_images


# Returns the k best images that match the query image
# weights for distance calculation, weights = [w11, w12, w21, w22]
def find_images(image_base, query_image_name, num_matches, level=4, long=False, mode="periodization", wave="db2",
                weights=[1, 1, 1, 1], wc1=1, wc2=1,
                wc3=1,
                threshold_dwt=0.5, threshold=70):
    start_time = time.perf_counter()
    # TODO: TEST remove mode and wave
    query_image = Image(image_name=query_image_name, level=level, long=long, mode=mode, wave=wave)
    # using additional dwt level check
    if long:
        distance_long(image_base, query_image, weights, wc1, wc2, wc3, threshold_dwt, threshold)
    else:
        distance_short(image_base, query_image, weights, wc1, wc2, wc3, threshold)
    # order based on distance
    image_base.sort(key=lambda img: img.distance)

    first_processed = 0
    # find first image that was processed (whose distance was computed) in sorted list
    for image in image_base:
        if image.distance >= 0:
            break
        first_processed += 1
    stop_time = time.perf_counter()
    compute_time = stop_time - start_time
    # if not enough images were processed, output random (or maybe throw exception in future)
    # TODO CHANGE BACK
    # if num_matches + first_processed - 1 >= len(image_base):
    #     extra = num_matches + first_processed - len(image_base)
    #     print("Not enough processed images, showing", extra, " unprocessed images.")
    #     return image_base[first_processed::] + image_base[0:extra:], compute_time
    if num_matches + first_processed - 1 >= len(image_base):
        extra = num_matches + first_processed - len(image_base)
        print("Not enough processed images, showing", extra, " unprocessed images.")
        return image_base[first_processed::], compute_time
    # otherwise extract the best matches
    else:
        return image_base[first_processed:first_processed + num_matches:], compute_time


# compute the distance of all the images in the image base from the query image, using both filtering based on standard
# deviation and level-5 filtering
def distance_long(image_base, query_image, weights, wc1, wc2, wc3, threshold_dwt, threshold):
    for image in image_base:
        image.filter_standard_deviation(query_image, threshold)
        # filter out images that were within the standard deviation
        # compute difference of level i+1 dwt components
        if image.distance != -1:
            image.image_distance(query_image, weights, wc1, wc2, wc3)
    max_distance = max(x.distance for x in image_base)
    # filter out images which are too far away in this coarser representation
    for image in image_base:
        if image.distance > threshold_dwt * max_distance:
            image.distance = -1
    # compute final distance based on level i dwt
    for image in image_base:
        if image.distance != -1:
            image.image_distance(query_image, weights, wc1, wc2, wc3, 5)


# compute the distance from all the images in the image base from the query image, using only filtering based on the
# standard deviation
def distance_short(image_base, query_image, weights, wc1, wc2, wc3, threshold):
    for image in image_base:
        image.filter_standard_deviation(query_image, threshold)
        # filter out images that were within the standard deviation
        # compute difference of level i dwt components
        if image.distance != -1:
            image.image_distance(query_image, weights, wc1, wc2, wc3)


# Calculate the average precision for the returned images
# Assuming the images are divided into semantic classes and that the class is denoted by the first letter/number of the
# image name
def calc_AP(query_name, returned_results, num_results):
    # use start number as class identifier
    start_letter = query_name[0]
    num_correct = 0
    for image in returned_results:
        if image.image_name[0] == start_letter:
            num_correct += 1
    return num_correct / num_results


def main():
    image_base = []
    if len(sys.argv) <= 1:
        print("Loading image base from json file")
        # image_base, time_taken = JsonParsing.load_from_json(
        #     "Json_Database/Corel1-k_db2_period_no_long.json")
        image_base, time_taken = JsonParsing.load_from_json(
            "/home/emily/Documents/2021/CSC5029Z/MiniProject/Json_Database/Corel1-k_db2_period_no_long.json")
        print("Feature base created. Time taken to create original feature base: ", time_taken)
    else:
        folder = sys.argv[1]
        # TODO: error if not image
        print("Loading image base...")
        count = 0
        # Timer for creating database of features
        start_time = time.perf_counter()
        for filename in os.listdir(folder):
            # computing level 4 dwt
            image_base.append(Image(filename))
            count += 1
            print(count)
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        print("Finished loading images. Time taken:", loading_time)

    while True:
        option = input("Enter a query image (or 'quit' to exit): ")
        if option == "quit":
            break
        else:
            matches = input("Enter number of matches to search for: ")
            if matches.isdigit():
                matches = int(matches)
                if matches > len(image_base):
                    print("Not enough images. Number of requested matches:", matches,
                          "Number of images to search: ",
                          len(image_base))
                else:
                    print("Beginning image retrieval")
                    matching_images, time_taken = find_images(image_base, option, matches)
                    print("Best matches found. Time taken:", time_taken)
                    print("Precision:", calc_AP(option, matching_images, matches))
                    view_image = input("View the best matching images (y\\n): ")
                    if view_image.lower() == "y":
                        show_images(option, matching_images)
                    save_image = input("Save the best matching images (y\\n): ")
                    if save_image.lower() == "y":
                        save_image_name = input("Enter name for image to be saved as: ")
                        save_images(option, matching_images, save_image_name + ".png")
                        print("Images saved to", save_image_name + ".png")
            else:
                print("Invalid match number specified:", matches, ".Enter an integer")


if __name__ == '__main__':
    main()
