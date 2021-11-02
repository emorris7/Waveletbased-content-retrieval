import os
import sys
import time

from ProcessImage import Image


# Returns the k best images that match the query image
# weights for distance calculation, weights = [w11, w12, w21, w22]
# TODO: TEST remove mode and wave
def find_images(image_base, query_image_name, num_matches, level, long, mode="periodization", wave="db2",
                weights=[1, 1, 1, 1], wc1=1, wc2=1,
                wc3=1,
                threshold_dwt=0.5, threshold=50):
    start_time = time.perf_counter()
    start_time_p = time.process_time()
    # TODO: TEST remove mode and wave
    query_image = Image(image_name=query_image_name, level=level, long=long, mode=mode, wave=wave)
    # print("Calculating distances")
    # using additional dwt level check
    if long:
        distance_long(image_base, query_image, weights, wc1, wc2, wc3, threshold_dwt, threshold)
    else:
        distance_short(image_base, query_image, weights, wc1, wc2, wc3, threshold)
    # order based on distance
    image_base.sort(key=lambda img: img.distance)

    # for image_prac in image_base:
    #     print(image_prac.image_name, image_prac.distance)

    first_processed = 0
    # find first image that was processed (whose distance was computed) in sorted list
    # print("Finding the ", num_matches, " best image matches")
    for image in image_base:
        if image.distance >= 0:
            break
        first_processed += 1
    # print(first_processed)
    stop_time = time.perf_counter()
    stop_time_p = time.process_time()
    compute_time = stop_time - start_time
    compute_time_p = stop_time_p - start_time_p
    # if not enough images were processed, output random (or maybe throw exception in future)
    # TODO CHANGE BACK
    # if num_matches + first_processed - 1 >= len(image_base):
    #     extra = num_matches + first_processed - len(image_base)
    #     print("Not enough processed images, showing", extra, " unprocessed images.")
    #     return image_base[first_processed::] + image_base[0:extra:], compute_time
    if num_matches + first_processed - 1 >= len(image_base):
        extra = num_matches + first_processed - len(image_base)
        print("Not enough processed images, showing", extra, " unprocessed images.")
        return image_base[first_processed::], compute_time, compute_time_p
    # otherwise extract the best matches
    else:
        return image_base[first_processed:first_processed + num_matches:], compute_time, compute_time_p


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


def distance_short(image_base, query_image, weights, wc1, wc2, wc3, threshold):
    for image in image_base:
        image.filter_standard_deviation(query_image, threshold)
        # filter out images that were within the standard deviation
        # compute difference of level i dwt components
        if image.distance != -1:
            image.image_distance(query_image, weights, wc1, wc2, wc3)


def calc_ARP(query_name, returned_results):
    # use start number as class identifier
    start_letter = query_name[0]
    num_correct = 0
    for image in returned_results:
        if image.image_name[0] == start_letter:
            num_correct += 1
    return num_correct / len(returned_results)


def main():
    if len(sys.argv) <= 1:
        print("Error: Must enter directory containing images to search")
    else:
        # image_base = load_for_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral.json")
        image_base = []
        folder = sys.argv[1]
        # TODO: error if not image
        print("Loading image base...")
        level = 4
        long = True
        count = 0
        # Timer for creating database of features
        start_time = time.perf_counter()
        for filename in os.listdir(folder):
            # computing level 4 dwt
            image_base.append(Image(filename, level, long))
            count += 1
            print(count)
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        print("Finished loading images. Time taken:", loading_time)
        # save_to_json("Coral.json", image_base, str(loading_time))
        print(len(image_base))

        while True:
            option = input("Enter a query image (or 'quit' to exit): ")
            if option == "quit":
                break
            else:
                # TODO: change beta threshold based on number of matches
                matches = eval(input("Enter number of matches to search for: "))
                if matches > len(image_base):
                    print("Not enough images. Number of requested matches:", matches, "Number of images to search: ",
                          len(image_base))
                else:
                    # print("Beginning image retrieval")
                    matching_images, time_taken = find_images(image_base, option, matches, level, long)
                    print("Time taken:", time_taken)
                    # print("Best matching images")
                    print(calc_ARP(option, matching_images))
                    # for img in matching_images:
                    #     print(img.image_name, img.distance)


if __name__ == '__main__':
    main()
