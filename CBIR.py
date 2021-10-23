import os
import sys

from ProcessImage import Image

# Returns the k best images that match the query image
def find_images(image_base, query_image_name, num_matches):
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
    for img in image_base:
        print(img.image_name, img.distance)
    first_processed = 0
    # find first image that was processed (whose distance was computed) in sorted list
    print("Finding the ", num_matches, " best image matches")
    for img in image_base:
        if img.distance >= 0:
            break
        first_processed += 1
    print(first_processed)
    # if not enough images were processed, output random (or maybe throw exception in future)
    if num_matches + first_processed - 1 >= len(image_base):
        extra = num_matches + first_processed - len(image_base)
        print("Not enough processed images, showing", extra, " unprocessed images.")
        return image_base[first_processed::] + image_base[0:extra:]
    # otherwise extract the best matches
    else:
        return image_base[first_processed:first_processed + num_matches:]


def main():
    if len(sys.argv) <= 1:
        print("Error: Must enter directory containing images to search")
    else:
        image_base = []
        folder = sys.argv[1]
        # TODO: error if not image
        print("Loading image base...")
        # count = 0
        for filename in os.listdir(folder):
            image_base.append(Image(filename))
            # count += 1
            # print(count)
        print("Finished loading images")
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
                    matching_images = find_images(image_base, option, matches)
                    print("Best matching images")
                    for img in matching_images:
                        print(img.image_name)


if __name__ == '__main__':
    main()
