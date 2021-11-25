# Waveletbased-content-retrieval
With the recent increase in visual data, the task of finding semantically related images based on
their content, known as content based image retrieval (CBIR), has become an increasingly important
one. This project, focuses on a particular approach to CBIR proposed by Wang et al (“Content-based image indexing and searching using Daubechies’ wavelets”). 

Two variations of the approach are explored. The first closely resembles the approach outlined in the paper. First the image is reduced to be of size 128x128. Then the image is transformed to a new set of colour axis, C_1, C_2 , C_3. A 4-level Discrete Wavelet Transform is then performed on the image in each of these color axis. The subbands and approximation image in level-4 for each color axis are then used as part of the feature for the image, along with the standard deviation of the approximation image. The standard deviation is used as part of the filtering process. The second approach utilizes PCA and KD-trees to optimize the initial approach for quicker and more accurate results. 
 
These approaches are applied to the Coral1-k dataset (http://wang.ist.psu.edu/docs/related/). Although this wavelet based approach performs well for some classes, it performs poorly overall achieving an average class precision of approximately 0.3. Results suggest that matching is done mainly based on colour.

# How to Run:
There are two different implementations that can be run, the basic implementation (CBIR.py/CBIRGUI.py) and the optimized implementation (PCAKD.py/PCAGUI.py). The implementations can be run through either a command line interface or GUI. Note that it is assumed that the image being searched for is in the /Coral1-k directory and that the first element in the images’ name determines which class it is semantically related to (this is done to compute an automatic precision result). While it is not necessary to follow this naming convention, the precision calculation will not be accurate without it.  The requirements.txt file can be used to set up a virtual environment and install the required packages.

Command Line usage:
Note- program_name = PCAKD.py or CBIR.py
1) To run the program with the precomputed and saved features (which will be loaded from a Json file), simply run: python3 program_name
2) To run the program and load and compute the images from scratch from a directory, run: python3 program_name <directory_name> 
3) Once the image features have been loaded, the user will be prompted to enter the name of an image to perform the query for.
4) Once the user has entered an image name, they will be prompted to enter a number of matches to search for.
5) Once the search is completed, the time taken for the query (in seconds) and the precision for the query is shown to the user and the user is given the option to view the matching images. If the user enters ‘y’, a window pops up displaying the matching images. Clicking any key on the keyboard will make this window disappear.
6) The user is then given the option to save the image. If they enter ‘y’, they are prompted to enter an file name for saving the image. The user is informed once the image has been saved.
7) The user is then given the option to search for another image, or quit the application by entering ‘quit’.

GUI Usage:
Note- program_name = PCAGUI.py or CBIRGUI.py
1) To run the program with the precomputed and saved features (which will be loaded from a Json file), simply run: python3 program_name
2) To run the program and load and compute the images from scratch from a directory, run: python3 program_name <directory_name> 
3) Once the image features have been loaded a screen will pop up with spaces for the user to enter a name of an query image and a number of images matches to find. Note that both boxes need to be filled and the box for the number of matches needs to contain and integer for the search to be performed.
4) Once the user enters the required information, they can press the ‘search’ button to begin the image query.
5) Once the query is complete, the user is shown the amount of time taken for the query (in seconds) and the precision for the query.
6) There are buttons presented to the user giving the they option to view the results, save the results and perform another query.
7) If the user chooses to save the results, they are presented with a space to enter a name for results to be saved under. Note that the suffix ‘.png’ will be automatically appended to the end of this. Once the user has entered a name, they can press the ‘save’ button to save the results. They are then once again shown the options to view the results or perform another query.
8) If the user chooses to view the results, a window pops up displaying the results. Pushing any button on the key board causes to window to close.
9) If the user chooses to perform another query, they are taken back to the original options.
10) To exit the program, the user can click on the red ‘x’ in the top right hand corner of the window
