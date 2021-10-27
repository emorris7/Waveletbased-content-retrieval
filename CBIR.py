import os
import sys
import time

from ProcessImage import Image


# Returns the k best images that match the query image
def find_images(image_base, query_image_name, num_matches):
    start_time = time.perf_counter()
    query_image = Image(query_image_name)
    # weights for distance calculation, weights = [w11, w12, w21, w22]
    # TODO: perhaps change weights
    weights = [1, 1, 1, 1]
    print("Calculating distances")
    for image in image_base:
        # other weights
        # TODO: change weights
        image.image_distance(query_image, weights, 1, 1, 1)
    image_base.sort(key=lambda img: img.distance)
    # for img in image_base:
    #     print(img.image_name, img.distance)
    first_processed = 0
    # find first image that was processed (whose distance was computed) in sorted list
    print("Finding the ", num_matches, " best image matches")
    for img in image_base:
        if img.distance >= 0:
            break
        first_processed += 1
    print(first_processed)
    stop_time = time.perf_counter()
    compute_time = stop_time - start_time
    # if not enough images were processed, output random (or maybe throw exception in future)
    if num_matches + first_processed - 1 >= len(image_base):
        extra = num_matches + first_processed - len(image_base)
        print("Not enough processed images, showing", extra, " unprocessed images.")
        return image_base[first_processed::] + image_base[0:extra:], compute_time
    # otherwise extract the best matches
    else:
        return image_base[first_processed:first_processed + num_matches:], compute_time


def distance_long(image_base, query_image, weights=[1, 1, 1, 1], wc1=1, wc2=1, wc3=1, threshold_dwt = 0.5, threshold=50):
    for image in image_base:
        # other weights
        # TODO: change weights
        image.image_distance(query_image, weights, wc1, wc2, wc3, threshold)

# def distance_short(image_base, query_image, weights=[1, 1, 1, 1], wc1=1, wc2=1, wc3=1, threshold=50):


def main():
    if len(sys.argv) <= 1:
        print("Error: Must enter directory containing images to search")
    else:
        # image_base = load_for_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral.json")
        image_base = []
        folder = sys.argv[1]
        # TODO: error if not image
        print("Loading image base...")
        count = 0
        # Timer for creating database of features
        start_time = time.perf_counter()
        for filename in os.listdir(folder):
            image_base.append(Image(filename))
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
                    print("Beginning image retrieval")
                    matching_images, time_taken = find_images(image_base, option, matches)
                    print("Time taken:", time_taken)
                    print("Best matching images")
                    for img in matching_images:
                        print(img.image_name, img.distance)


if __name__ == '__main__':
    main()
