import os
import sys
import time

from CBIR import find_images
from JsonParsing import load_from_json
from ProcessImage import Image


# Find best wavelet
def wavelet_test(folder):
    wavelets = ["db9", "db10"]
    modes = ["symmetric", "periodization"]
    for m in modes:
        for w in wavelets:
            # for Coral 1-k
            total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

            level = 4
            long = True
            image_base = []
            count = 0
            start_time = time.perf_counter()
            for filename in os.listdir(folder):
                # computing level 4 dwt
                image_base.append(Image(image_name=filename, level=level, long=long, mode=m, wave=w))
                count += 1
                if count % 100 == 0:
                    print(count)
            end_time = time.perf_counter()
            loading_time = end_time - start_time

            count = 0
            for image in image_base:
                name = image.image_name
                found, time_taken = find_images(image_base=image_base, query_image_name=name, num_matches=100,
                                                level=level, long=long, mode=m, wave=w, threshold_dwt=1)
                precision = calc_rp(name, found)
                total[name[0]] += precision
                max_dict[name[0]] = max(precision, max_dict[name[0]])
                min_dict[name[0]] = min(precision, min_dict[name[0]])
                count += 1
                if count % 100 == 0:
                    print(count)

            file = open(w + "_" + m + ".txt", 'w')
            print("Finished loading images. Time taken:", loading_time)
            total_average = 0
            file.write("Time:" + str(loading_time) + "\n")
            for key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                average = total[key] / 100
                total_average += average
                out_str = "Class: " + key + " min: " + str(min_dict[key]) + " max: " + str(
                    max_dict[key]) + " average: " + str(average)
                file.write(out_str + "\n")
                print("Class", key)
                print("Min:", min_dict[key])
                print("Max:", max_dict[key])
                print("Average:", average)
            file.write(str(total_average / 10) + "\n")
            file.close()


def no_level_test(folder):
    thresholds = [50, 60, 70, 80, 90]
    image_base = load_from_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/Corel1-k_db2_period_no_long.json")
    # parameters
    num_iterations = 5
    num_per_class = 11
    level = 4
    long = False
    count = 0

    for t in thresholds:
        total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

        time_total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        max_time = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        min_time = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}

        time_total_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        max_time_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        min_time_p = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}
        for filename in os.listdir(folder):
            # throw away first 5 runs
            for i in range(1):
                found, time_taken, time_taken_p = find_images(image_base=image_base, query_image_name=filename,
                                                              num_matches=100,
                                                              level=level, long=long,
                                                              threshold=t)
            # run 25 times and take average
            final_found = []
            total_p = 0
            total_np = 0

            for i in range(num_iterations):
                found, time_taken, time_taken_p = find_images(image_base=image_base, query_image_name=filename,
                                                              num_matches=100,
                                                              level=level, long=long,
                                                              threshold=t)
                final_found = found
                total_p += time_taken_p
                total_np += time_taken

            # record precision
            precision = calc_rp(filename, final_found)
            total[filename[0]] += precision
            max_dict[filename[0]] = max(precision, max_dict[filename[0]])
            min_dict[filename[0]] = min(precision, min_dict[filename[0]])

            # record perf_timer
            average_perf = total_np / num_iterations
            time_total[filename[0]] += average_perf
            max_time[filename[0]] = max(average_perf, max_time[filename[0]])
            min_time[filename[0]] = average_perf if min_time[filename[0]] == -1 else min(average_perf,
                                                                                         min_time[filename[0]])

            # record process_timer
            average_process = total_p / num_iterations
            time_total_p[filename[0]] += average_process
            max_time_p[filename[0]] = max(average_process, max_time_p[filename[0]])
            min_time_p[filename[0]] = average_process if min_time_p[filename[0]] == -1 else min(average_process,
                                                                                                min_time_p[
                                                                                                    filename[0]])
            count += 1
            if count % 50 == 0:
                print(count)

        # write results to a file
        file = open(str(t) + "_" + "one_level_4.txt", 'w')
        total_average = 0
        total_average_time = 0
        total_average_time_p = 0
        for key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            average = total[key] / num_per_class
            average_time = time_total[key] / num_per_class
            average_time_p = time_total_p[key] / num_per_class
            # for calculating the final average time
            total_average += average
            total_average_time += average_time
            total_average_time_p += average_time_p
            # string for class precision
            out_str = "Class: " + key + " min: " + str(min_dict[key]) + " max: " + str(
                max_dict[key]) + " average: " + str(average)
            # string for time with perf_counter
            out_str_2 = "Time: min: " + str(min_time[key]) + " max: " + str(
                max_time[key]) + " average: " + str(average_time)
            # string for time with process_time
            out_str_3 = "Time: min: " + str(min_time_p[key]) + " max: " + str(
                max_time_p[key]) + " average: " + str(average_time_p)
            file.write(out_str + "\n")
            file.write(out_str_2 + "\n")
            file.write(out_str_3 + "\n")
            print("Class", key)
            print("Min:", min_dict[key])
            print("Max:", max_dict[key])
            print("Average:", average)
        file.write(str(total_average / 10) + "\n")
        file.write(str(total_average_time / 10) + "\n")
        file.write(str(total_average_time_p / 10) + "\n")
        file.close()


def test_thresholds(folder):
    dwt_threshold = [1, 0.2, 0.4, 0.6, 0.8]
    # dwt_threshold = [1]
    threshold = [50, 60, 70, 80, 90]
    # threshold = [60, 70, 80, 90]

    # parameters
    num_iterations = 5
    # using subset
    num_per_class = 11
    level = 4
    long = True
    count = 0

    image_base = load_from_json("/home/emily/Documents/2021/CSC5029Z/MiniProject/Corel1-k_db2_period_long.json")
    for t in threshold:
        for dwtt in dwt_threshold:
            total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

            time_total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            max_time = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            min_time = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}

            time_total_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            max_time_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            min_time_p = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}

            for filename in os.listdir(folder):
                # throw away first 5 runs
                for i in range(1):
                    found, time_taken, time_taken_p = find_images(image_base=image_base, query_image_name=filename,
                                                                  num_matches=100,
                                                                  level=level, long=long, threshold_dwt=dwtt,
                                                                  threshold=t)
                # run 25 times and take average
                final_found = []
                total_p = 0
                total_np = 0

                for i in range(num_iterations):
                    found, time_taken, time_taken_p = find_images(image_base=image_base, query_image_name=filename,
                                                                  num_matches=100,
                                                                  level=level, long=long, threshold_dwt=dwtt,
                                                                  threshold=t)
                    final_found = found
                    total_p += time_taken_p
                    total_np += time_taken

                # record precision
                precision = calc_rp(filename, final_found)
                total[filename[0]] += precision
                max_dict[filename[0]] = max(precision, max_dict[filename[0]])
                min_dict[filename[0]] = min(precision, min_dict[filename[0]])

                # record perf_timer
                average_perf = total_np / num_iterations
                time_total[filename[0]] += average_perf
                max_time[filename[0]] = max(average_perf, max_time[filename[0]])
                min_time[filename[0]] = average_perf if min_time[filename[0]] == -1 else min(average_perf,
                                                                                             min_time[filename[0]])

                # record process_timer
                average_process = total_p / num_iterations
                time_total_p[filename[0]] += average_process
                max_time_p[filename[0]] = max(average_process, max_time_p[filename[0]])
                min_time_p[filename[0]] = average_process if min_time_p[filename[0]] == -1 else min(average_process,
                                                                                                    min_time_p[
                                                                                                        filename[0]])
                count += 1
                if count % 50 == 0:
                    print(count)

            # write results to a file
            file = open(str(dwtt) + "_" + str(t) + ".txt", 'w')
            total_average = 0
            total_average_time = 0
            total_average_time_p = 0
            for key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                average = total[key] / num_per_class
                average_time = time_total[key] / num_per_class
                average_time_p = time_total_p[key] / num_per_class
                # for calculating the final average time
                total_average += average
                total_average_time += average_time
                total_average_time_p += average_time_p
                # string for class precision
                out_str = "Class: " + key + " min: " + str(min_dict[key]) + " max: " + str(
                    max_dict[key]) + " average: " + str(average)
                # string for time with perf_counter
                out_str_2 = "Time: min: " + str(min_time[key]) + " max: " + str(
                    max_time[key]) + " average: " + str(average_time)
                # string for time with process_time
                out_str_3 = "Time: min: " + str(min_time_p[key]) + " max: " + str(
                    max_time_p[key]) + " average: " + str(average_time_p)
                file.write(out_str + "\n")
                file.write(out_str_2 + "\n")
                file.write(out_str_3 + "\n")
                print("Class", key)
                print("Min:", min_dict[key])
                print("Max:", max_dict[key])
                print("Average:", average)
            file.write(str(total_average / 10) + "\n")
            file.write(str(total_average_time / 10) + "\n")
            file.write(str(total_average_time_p / 10) + "\n")
            file.close()


def test_wavelet(folder):
    wavelet = ["db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10"]
    mode = ["symmetric", "periodization"]

    level = 4
    long = False
    num_per_class = 100

    for m in mode:
        for w in wavelet:
            total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
            min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

            image_base = []
            count = 0
            print(m, w)
            start_time = time.perf_counter()
            for filename in os.listdir(folder):
                # computing level 4 dwt
                image_base.append(Image(image_name=filename, level=level, long=long, mode=m, wave=w))
                count += 1
                if count % 100 == 0:
                    print(count)
            end_time = time.perf_counter()
            loading_time = end_time - start_time

            for filename in os.listdir(folder):
                found, time_taken, time_taken_p = find_images(image_base=image_base, query_image_name=filename,
                                                              num_matches=100,
                                                              level=level, long=long, mode=m, wave=w,
                                                              threshold=50)

                # record precision
                precision = calc_rp(filename, found)
                total[filename[0]] += precision
                max_dict[filename[0]] = max(precision, max_dict[filename[0]])
                min_dict[filename[0]] = min(precision, min_dict[filename[0]])

                # write results to a file
            file = open(w + "_" + m + ".txt", 'w')
            total_average = 0
            file.write("Time taken: " + str(loading_time) + "\n")
            for key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                average = total[key] / num_per_class
                total_average += average
                # string for class precision
                out_str = "Class: " + key + " min: " + str(min_dict[key]) + " max: " + str(
                    max_dict[key]) + " average: " + str(average)
                file.write(out_str + "\n")
                print("Class", key)
                print("Min:", min_dict[key])
                print("Max:", max_dict[key])
                print("Average:", average)
            file.write(str(total_average / 10) + "\n")
            file.close()


def test_precision(folder):
    level = 4
    long = False
    image_base = []
    count = 0
    for filename in os.listdir(folder):
        # computing level 4 dwt
        image_base.append(Image(image_name=filename, level=level, long=long))
        count += 1
        if count % 100 == 0:
            print(count)
    print("Finished loading")

    count = 0
    num_per_class = 100
    num_matches = 20
    total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

    for filename in os.listdir(folder):
        found, time_taken, time_taken_p = find_images(image_base=image_base,
                                                      query_image_name=filename, num_matches=num_matches, level=level,
                                                      long=long)

        # record precision
        precision = calc_rp(filename, found, num_matches)
        total[filename[0]] += precision
        max_dict[filename[0]] = max(precision, max_dict[filename[0]])
        min_dict[filename[0]] = min(precision, min_dict[filename[0]])

        count += 1
        if count % 50 == 0:
            print(count)

    # write results to a file
    file = open("basic_precision_20.txt", 'w')
    total_average = 0
    for key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        average = total[key] / num_per_class
        # for calculating the final average time
        total_average += average
        # string for class precision
        out_str = "Class: " + key + " min: " + str(min_dict[key]) + " max: " + str(
            max_dict[key]) + " average: " + str(average)
        file.write(out_str + "\n")
        print("Class", key)
        print("Min:", min_dict[key])
        print("Max:", max_dict[key])
        print("Average:", average)
    file.write(str(total_average / 10) + "\n")
    file.close()


def calc_rp(query_name, returned_results, num_matches):
    # use start number as class identifier
    start_letter = query_name[0]
    num_correct = 0
    for image in returned_results:
        if image.image_name[0] == start_letter:
            num_correct += 1
    return num_correct / num_matches


def main():
    if len(sys.argv) <= 1:
        print("Requires directory for testing")
    else:

        folder = sys.argv[1]
        # test_thresholds(folder)
        # no_level_test(folder)
        # test_wavelet(folder)
        test_precision(folder)


if __name__ == '__main__':
    main()
