import os
import math
import matplotlib.pyplot as plt
import numpy as np


def get_color(c, x, Nc):  # Extracted from YOLOv3(image.c)
    colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = (x / Nc) * 5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return r


def graphics_representation(list_of_edges, path_results, file_classes_names):

    # Load the file classes.names
    labelsPath = "Resources/" + file_classes_names
    labels = open(labelsPath).read().strip().split("\n")

    for value in list_of_edges:
        #####################################################################
        # First: read the distance from ref1 txt and transversal distance txt
        #####################################################################
        matrix_data_d_ref1 = np.loadtxt("./Resources/EvalWeights/Edges_d_from_ref1/"+"Edge_"+str(value)+".txt")
        matrix_data_d_tran = np.loadtxt("./Resources/EvalWeights/Transversal_dist/"+"EdgeTrans_"+str(value)+".txt")

        # Check if only have one column the txt data
        try:
            if matrix_data_d_tran.shape[1] != 0 and matrix_data_d_ref1.shape[1] != 0:
                pass
        except:
            # Only have one column so is necessary a reshape
            matrix_data_d_ref1 = np.array([matrix_data_d_ref1]).T
            matrix_data_d_tran = np.array([matrix_data_d_tran]).T

        #########################
        # Second: draw the values
        #########################
        # Create tha auxiliar list with classes
        classes_det = []

        index_class = 0
        Number_classes = 12
        # Auxiliar variable for scale the y-axis
        max_d_ref1 = 0.0
        max_d_trans = 0.0

        for longitudinal, transversal in zip(matrix_data_d_ref1, matrix_data_d_tran):

            flag = False
            # Define the color of representation
            offset = (index_class * 123457) % Number_classes

            red = get_color(2, offset, Number_classes)
            green = get_color(1, offset, Number_classes)
            blue = get_color(0, offset, Number_classes)
            color_class = [red, green, blue]

            # Draw each value of list class object
            for value_long, value_tran in zip(longitudinal, transversal):
                if math.isnan(value_long) and math.isnan(value_tran):
                    break
                else:
                    # Update the variable show_label to True
                    show_label = True
                    # Update the max value for d_from_ref1
                    if max_d_ref1 < value_long:
                        max_d_ref1 = value_long
                    # Update the max value for abs(d_trans)
                    if max_d_trans < abs(value_tran):
                        max_d_trans = abs(value_tran)

                    if not flag:
                        # controls if it corresponds to the first representation of a class (flag=False)
                        # or not (flag=True)
                        plt.scatter(x=value_tran, y=value_long, s=300, color=color_class, edgecolors='black', alpha=1)
                        flag = True
                        classes_det.append(labels[index_class])
                    else:
                        plt.scatter(x=value_tran, y=value_long, s=300, color=color_class, edgecolors='black', alpha=1,
                                    label='_nolegend_')

                    plt.plot((value_tran, 0), (value_long, value_long), 'b--', linewidth=1, label='_nolegend_')

            # Increment the index object class
            index_class += 1

        # Update the limits and labels of axis
        plt.ylim(0.0, max_d_ref1 + 2.0)
        plt.xlim(-max_d_trans - 2.0, max_d_trans + 2.0)
        plt.xticks(np.arange(np.floor(-max_d_trans - 1), np.ceil(max_d_trans + 2), step=1), fontsize=20)
        plt.yticks(np.arange(0.0, max_d_ref1, step=3), fontsize=20)
        # Show the grid
        plt.grid()
        # Plot the Line of Edge
        plt.plot((0, 0), (0, max_d_ref1), 'b', linewidth=5)
        # Show the legend
        plt.legend(classes_det, fontsize=14, loc='upper left')

        # Save the figure
        plt.savefig(path_results + "Edge_" + str(value) + ".png", bbox_inches='tight')
        plt.cla()


def main(path_results, path_resources_tests):
    """
    path_results: path for save the map from annotation
    path_resources_tests: path for extract the edges necessary for create the map annotation
    """
    # Check if exists or not the directory where will be saved the results annotation from data test
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Extract the edges for annotation
    # First: load the matrix edges
    matrix_edges = np.loadtxt("./Resources/EvalWeights/Matrix_Edges.txt")
    # Second: extract the edges with the nodes from directories test
    list_edges = []
    for directory in os.listdir(path_resources_tests):
        split_directory = directory.split("N")
        init_node = int(split_directory[1])
        dest_node = int(split_directory[2])
        list_edges.append(int(matrix_edges[init_node-1][dest_node-1]))

    # Third: call the function that create the map annotation
    graphics_representation(list_of_edges=list_edges,
                            path_results=path_results,
                            file_classes_names="classes.names")


if __name__ == "__main__":
    main(path_results="./Results/Annotation_Map_Results/",
         path_resources_tests="./Resources/Test/")
