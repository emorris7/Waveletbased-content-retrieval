import os
import sys
import time
import tkinter as tk

import CBIR
import JsonParsing
import ProcessImage

image_base = []
matching_images = []
img_name = ""


def main():
    global image_base
    if len(sys.argv) <= 1:
        print("Loading image base from json file")
        image_base, time_taken = JsonParsing.load_from_json(
            "Json_Database/Corel1-k_db2_period_no_long.json")
        print("Feature base created. Time taken to create original feature base: ", time_taken)
    else:
        folder = sys.argv[1]
        print("Loading image base...")
        count = 0
        # Timer for creating database of features
        start_time = time.perf_counter()
        for filename in os.listdir(folder):
            # computing level 4 dwt
            image_base.append(ProcessImage.Image(filename))
            count += 1
            print(count)
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        print("Finished loading images. Time taken:", loading_time)

    window = tk.Tk()
    window.geometry("600x600")
    welcome_label = tk.Label(
        text="WELCOME TO WB-CBIR",
        foreground="white",  # Set the text color to white
        background="black",  # Set the background color to black
        width=80,
        height=5,
        font=("Arial", 20)
    )

    instruct_label = tk.Label(text="Enter image to search for:", foreground="black",  # Set the text color to white
                              background="white",  # Set the background color to black
                              width=40,
                              height=5,
                              font=("Arial", 10))
    match_label = tk.Label(text="Enter the number of matches to return:", foreground="black",
                           # Set the text color to white
                           background="white",  # Set the background color to black
                           width=40,
                           height=5,
                           font=("Arial", 10))

    def pack_search():
        welcome_label.pack()
        instruct_label.pack()
        entry.pack(pady=20)
        match_label.pack()
        entry_match.pack(pady=20)
        search_image.pack()

    def unpack_search():
        welcome_label.pack_forget()
        instruct_label.pack_forget()
        entry.pack_forget()
        match_label.pack_forget()
        entry_match.pack_forget()
        search_image.pack_forget()

    def pack_display():
        query_result_label.pack()
        time_label.pack(pady=20)
        precision_label.pack()
        save.pack(pady=20)
        view.pack()
        search_again.pack(pady=20)

    def unpack_display():
        query_result_label.pack_forget()
        time_label.pack_forget()
        precision_label.pack_forget()
        save.pack_forget()
        view.pack_forget()
        search_again.pack_forget()

    def search_click(event):
        global img_name
        global matching_images
        img_name = entry.get()
        number = entry_match.get()
        if number.isdigit() and img_name != "":
            # put searching stuff here
            matching_images, time_taken = CBIR.find_images(image_base, img_name, int(number))
            time_var.set("Time taken for query: " + "{:.5f}".format(time_taken))
            precision_var.set("Precision for image query: " + str(CBIR.calc_AP(img_name, matching_images, int(number))))
            # set up graphics
            unpack_search()
            query_var.set("QUERY RESULTS FOR IMAGE: " + img_name)
            pack_display()

    def save_click(event):
        save.pack_forget()
        view.pack_forget()
        search_again.pack_forget()
        filename_label.pack(pady=20)
        enter_file.pack()
        final_save.pack(pady=20)

    def final_save_click(event):
        filename = enter_file.get()
        if filename != "":
            enter_file.delete(0, 'end')
            ProcessImage.save_images(img_name, matching_images, filename + ".png")
            view.pack(pady=20)
            search_again.pack()
            final_save.pack_forget()
            enter_file.pack_forget()
            filename_label.pack_forget()

    def view_click(event):
        ProcessImage.show_images(img_name, matching_images)

    def search_again_click(event):
        # put query info back
        unpack_display()
        entry.delete(0, 'end')
        entry_match.delete(0, 'end')
        pack_search()

    entry = tk.Entry()
    entry_match = tk.Entry()

    search_image = tk.Button(text="Search")
    search_image.bind("<Button-1>", search_click)
    pack_search()

    # after searching
    query_var = tk.StringVar()
    query_result_label = tk.Label(
        textvariable=query_var,
        foreground="white",  # Set the text color to white
        background="black",  # Set the background color to black
        width=80,
        height=5,
        font=("Arial", 20)
    )
    time_var = tk.StringVar()
    time_label = tk.Label(textvariable=time_var, background="white",  # Set the background color to black
                          width=40,
                          height=5,
                          font=("Arial", 10))
    precision_var = tk.StringVar()
    precision_label = tk.Label(textvariable=precision_var, background="white",  # Set the background color to black
                               width=40,
                               height=5,
                               font=("Arial", 10))
    save = tk.Button(text="Save Results")
    save.bind("<Button-1>", save_click)
    view = tk.Button(text="View Results")
    view.bind("<Button-1>", view_click)
    search_again = tk.Button(text="Search")
    search_again.bind("<Button-1>", search_again_click)

    filename_label = tk.Label(text="Enter filename to save results to:", background="white",
                              # Set the background color to black
                              width=40,
                              height=5,
                              font=("Arial", 10))
    enter_file = tk.Entry()
    final_save = tk.Button(text="Save")
    final_save.bind("<Button-1>", final_save_click)

    window.mainloop()


if __name__ == '__main__':
    main()
