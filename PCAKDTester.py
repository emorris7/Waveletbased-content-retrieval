import os
import time

from sklearn.neighbors import KDTree

from PCAKD import extract_features, pca_database, find_images, find_images_kd
from ProcessImage import PCAImage


def test_PCA(folder_load, folder_search):
    level = 4
    long = False
    # start timers
    start_time = time.perf_counter()
    start_time_p = time.process_time()
    data_array, image_names = extract_features(folder_load, level, long)
    stand_scaler, pca_images, reduced_data = pca_database(data_array, 0.6)
    image_base = []
    for i in range(len(image_names)):
        image_base.append(PCAImage(image_names[i], reduced_data[i]))
    # calculate time taken
    stop_time = time.perf_counter()
    stop_time_p = time.process_time()
    compute_time = stop_time - start_time
    compute_time_p = stop_time_p - start_time_p
    print("Time taken to load:", compute_time, compute_time_p)

    num_iterations = 5
    count = 0
    num_per_class = 11
    total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

    time_total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_time = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_time = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}

    time_total_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_time_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_time_p = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}
    for filename in os.listdir(folder_search):
        # throw away first 5 runs
        for i in range(1):
            found, time_taken, time_taken_p = find_images(stand_scaler=stand_scaler, pca_images=pca_images,
                                                          image_base=image_base, query_image_name=filename,
                                                          num_matches=100,
                                                          level=level, long=long)
        # run 25 times and take average
        final_found = []
        total_p = 0
        total_np = 0

        for i in range(num_iterations):
            found, time_taken, time_taken_p = find_images(stand_scaler=stand_scaler, pca_images=pca_images,
                                                          image_base=image_base, query_image_name=filename,
                                                          num_matches=100,
                                                          level=level, long=long)
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
        print(count)

        # write results to a file
    file = open("0.6_PCA.txt", 'w')
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


def test_KD(folder_load, folder_search):
    level = 4
    long = False
    # start timers
    start_time = time.perf_counter()
    start_time_p = time.process_time()
    data_array, image_names = extract_features(folder_load, level, long)
    stand_scaler, pca_images, reduced_data = pca_database(data_array)
    # https: // scikit - learn.org / stable / modules / generated / sklearn.neighbors.KDTree.html
    tree = KDTree(reduced_data, leaf_size=2)
    # calculate time taken
    stop_time = time.perf_counter()
    stop_time_p = time.process_time()
    compute_time = stop_time - start_time
    compute_time_p = stop_time_p - start_time_p
    print("Time taken to load:", compute_time, compute_time_p)

    num_iterations = 5
    count = 0
    num_per_class = 11
    total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

    time_total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_time = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_time = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}

    time_total_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_time_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_time_p = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}
    for filename in os.listdir(folder_search):
        # throw away first 5 runs
        for i in range(1):
            found, time_taken, time_taken_p = find_images_kd(stand_scaler=stand_scaler, pca_images=pca_images,
                                                             kd_tree=tree, image_names=image_names,
                                                             query_image_name=filename, num_matches=100, level=level,
                                                             long=long)
        # run 25 times and take average
        final_found = []
        total_p = 0
        total_np = 0

        for i in range(num_iterations):
            found, time_taken, time_taken_p = find_images_kd(stand_scaler=stand_scaler, pca_images=pca_images,
                                                             kd_tree=tree, image_names=image_names,
                                                             query_image_name=filename, num_matches=100, level=level,
                                                             long=long)
            final_found = found
            total_p += time_taken_p
            total_np += time_taken

        # record precision
        precision = calc_rp_kd(filename, final_found)
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
        print(count)

        # write results to a file
    file = open("KD_PCA.txt", 'w')
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


def test_KD_leaf_size(folder_load, folder_search):
    level = 4
    long = False
    leaves = [2, 6, 10, 14, 18]
    # start timers
    data_array, image_names = extract_features(folder_load, level, long)
    stand_scaler, pca_images, reduced_data = pca_database(data_array)
    # https: // scikit - learn.org / stable / modules / generated / sklearn.neighbors.KDTree.html
    for leaf in leaves:
        tree = KDTree(reduced_data, leaf_size=leaf)
        # calculate time taken
        print("Finished load")

        num_iterations = 5
        count = 0
        num_per_class = 11
        total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

        time_total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        max_time = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        min_time = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}

        time_total_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        max_time_p = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        min_time_p = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1, "8": -1, "9": -1}
        for filename in os.listdir(folder_search):
            # throw away first 5 runs
            for i in range(1):
                found, time_taken, time_taken_p = find_images_kd(stand_scaler=stand_scaler, pca_images=pca_images,
                                                                 kd_tree=tree, image_names=image_names,
                                                                 query_image_name=filename, num_matches=100,
                                                                 level=level,
                                                                 long=long)
            # run 25 times and take average
            final_found = []
            total_p = 0
            total_np = 0

            for i in range(num_iterations):
                found, time_taken, time_taken_p = find_images_kd(stand_scaler=stand_scaler, pca_images=pca_images,
                                                                 kd_tree=tree, image_names=image_names,
                                                                 query_image_name=filename, num_matches=100,
                                                                 level=level,
                                                                 long=long)
                final_found = found
                total_p += time_taken_p
                total_np += time_taken

            # record precision
            precision = calc_rp_kd(filename, final_found)
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
            print(count)

            # write results to a file
        file = open("KD_PCA" + str(leaf) + ".txt", 'w')
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


def test_precision(folder):
    level = 4
    long = False
    data_array, image_names = extract_features(folder, level, long)
    stand_scaler, pca_images, reduced_data = pca_database(data_array)
    # https: // scikit - learn.org / stable / modules / generated / sklearn.neighbors.KDTree.html
    tree = KDTree(reduced_data, leaf_size=2)
    print("Finished loading")

    count = 0
    num_per_class = 100
    num_matches = 20
    total = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    max_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
    min_dict = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}

    for filename in os.listdir(folder):
        found, time_taken, time_taken_p = find_images_kd(stand_scaler=stand_scaler, pca_images=pca_images,
                                                         kd_tree=tree, image_names=image_names,
                                                         query_image_name=filename, num_matches=num_matches, level=level,
                                                         long=long)

        # record precision
        precision = calc_rp_kd(filename, found, num_matches)
        total[filename[0]] += precision
        max_dict[filename[0]] = max(precision, max_dict[filename[0]])
        min_dict[filename[0]] = min(precision, min_dict[filename[0]])

        count += 1
        if count % 50 == 0:
            print(count)

    # write results to a file
    file = open("KD_PCA_precision_20.txt", 'w')
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


def calc_rp(query_name, returned_results):
    # use start number as class identifier
    start_letter = query_name[0]
    num_correct = 0
    for image in returned_results:
        if image.image_name[0] == start_letter:
            num_correct += 1
    return num_correct / 100


def calc_rp_kd(query_name, returned_results, num_matches):
    # use start number as class identifier
    start_letter = query_name[0]
    num_correct = 0
    for image in returned_results:
        if image[0] == start_letter:
            num_correct += 1
    return num_correct / num_matches


def main():
    folder_load = "/home/emily/Documents/2021/CSC5029Z/MiniProject/Coral1-k"
    folder_search = "/home/emily/Documents/2021/CSC5029Z/MiniProject/subsets"
    test_precision(folder_load)
    #test_KD_leaf_size(folder_load, folder_search)


if __name__ == '__main__':
    main()
