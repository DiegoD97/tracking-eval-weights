########################################################################################################################
# FILE WITH THE CLASS DETECTION
########################################################################################################################

# LIBRARIES TO IMPORT
import math
import numpy as np
import os
import subprocess
import copy
import statistics
import cv2
import matplotlib.pyplot as plt
import node_classification


class Detection:
    #################################################################
    # BUILDER OF THE CLASS
    #################################################################
    def __init__(self, file_ObjectClasses, version_YOLO):
        # CHECK THE YOLO VERSION
        self.coord_hor, self.coord_depth = self.check_vYOLO(version_YOLO)
        # LOAD THE OBJECT-CLASS LABELS OUR YOLO MODEL WAS TRAINED ON
        try:
            labelsPath = "./Resources/" + file_ObjectClasses
            self.labels = open(labelsPath).read().strip().split("\n")
            self.n_classes = len(self.labels)  # Number of classes of objects
        except:
            print("File " + file_ObjectClasses + " is not found")

        # PARAMETERS FOR THE CAMERA
        # self.aperture_cam = 2 * math.atan(2.88/5.765)   # Camera aperture for viewing angle (radians)
        self.aperture_cam = 69.0 * math.pi / 180.0
        self.aperture = self.aperture_cam / 2           # Half value of the aperture angle of view's camera (radians)

        # PARAMETERS FOR THE IMAGE CAPTURATED
        self.height = 480   # In pixel
        self.width = 640    # In pixel

        # VARIABLES FOR TRACKING
        self.objects_estimation = dict((key, {'d_from_ref1': [], 'd_trans': [], 'u_clust': 0.0}) for key in self.labels)
        # Complete the data for umbral clustering for each class
        labelsPathClust = "./Resources/umbral_clust.txt"
        labels_clust = open(labelsPathClust).read().strip().split("\n")
        for key_class, u_clust_val in zip(self.labels, labels_clust):
            self.objects_estimation[key_class]['u_clust'] = float(u_clust_val)

        self.list_results = []
        self.list_YOLO_Camera_results = []
        self.dict_YOLO_Camera = {
            'xc_objects': None,
            'yc_objects': None,
            'depth_objects': None
        }

        ##########################################################################################
        # DECLARACIONES DE LOS NODOS, LOS TOPICS PUBLISHER Y SUBSCRIBER CUANDO SE INTEGRE EN ROS
        ##########################################################################################

        ##########################################################################################

    ###################################################################################################
    # FUNCTION FOR CHECKING THE VERSION OF YOLO
    ###################################################################################################

    def check_vYOLO(self, version):
        # In case to use the first version, the depth is considerate
        if version == 'YOLO_v1':
            return 2, 1
        # In otherwise, the depth is not considerate
        elif version == 'YOLO_v2':
            return 1, None

    ##################################################################################################
    # FUNCTION FOR FILTER THE DATA DEPTH
    ##################################################################################################
    @staticmethod
    def filter_data_depth_before_median(array_depth):
        array_depth_aux = []
        for value in array_depth[0]:
            if value < 1.0 or value > 10.0:
                pass
            else:
                array_depth_aux.append(value)

        # With the array results after to be filtered, now is necessary comprobate
        # the dimensions between array_depth_aux (results after to be filtered) and
        # array_depth (the original array) and if the error between dimension's
        # array is more 20 %, return a cero array for be deleted more later
        error_dimensions = abs(len(array_depth[0]) - len(array_depth_aux)) / len(array_depth[0])
        if error_dimensions > 0.2:
            # The array is empty so return a cero array
            array_depth_aux = [0]

        return np.array(array_depth_aux)

    ###################################################################################################
    # FUNCTION FOR SHOW THE DISTRIBUTTION FROM DEPTH DATA
    ###################################################################################################
    @staticmethod
    def show_bounding_depth(bounding_depth, yolo_depth_name, label_obj):
        aux_list_data = list(bounding_depth[0])
        aux_list_data.sort()
        aux_index_list = list(range(len(aux_list_data)))
        plt.plot(aux_index_list, aux_list_data)
        plt.xlabel("pixels")
        plt.ylabel("Value Depth")
        plt.title(yolo_depth_name+'\n'+label_obj)
        plt.show()

    ###################################################################################################
    # FUNCTION THAT READS THE OUTPUT DATA FROM YOLO
    ###################################################################################################

    def read_CoordYOLO(self, YOLO_BoundingBox, YOLO_depth=None):

        """
        This function read the output data created with the CNN YOLO. The data contain the object class
        and the estimation of the centroid coordinates (x,y) in pixel from the image.
        :param YOLO_BoundingBox: this is a .txt file with the coordinates of bounding box of objects
        :param YOLO_depth: this is a .txt with the data depth from YOLO
        :return:
        """
        # Init the var 'depth' to false. In this version is considerate the depth of objects from images
        depth = False

        # Obtain the horizontal coordinates from detected objects and their depth
        xcoord_object_detected = [None] * self.n_classes           # Coordinate horizontal from detected objects
        ycoord_object_detected = [None] * self.n_classes           # Coordinate vertical from detected objects
        depth_object_detected = [None] * self.n_classes            # Depth from the detected objects

        # Open and read the file where is saved the coordinates of bounding box from objects detected by YOLO
        f = open(YOLO_BoundingBox, 'r')

        # In case to have the depth of the object is considerate, the previous step is read the .txt file
        if YOLO_depth is not None:
            """
            # Open the file
            f_depth = open(YOLO_depth, 'r')
            # Auxiliar vars: matrix depth and bounding depth
            matrix_depth = []
            # Loop for complete the matrix depth
            for d in f_depth:
                row = d.split(' ')
                # Steps for delete the '\n' from last value from list
                resultado = row[-1]
                resultado = resultado[0:4]
                row[-1] = resultado
                # Append the results
                matrix_depth.append(row)

            matrix_depth = np.array(matrix_depth)
            matrix_depth = matrix_depth.astype(float)
            """
            matrix_depth = np.loadtxt(YOLO_depth)

        # Loop for saved in the vars the data
        for x in f:
            line = x.split(' ')
            if float(line[1]) < 0.8:
                pass
            else:
                # Centroid of objects
                xc = int(float(line[2]) + 0.5 * (float(line[4]) - float(line[2])))
                yc = int(float(line[3]) + 0.5 * (float(line[5]) - float(line[3])))

                # In case to have the depth of the object is considerate
                if YOLO_depth is not None:
                    depth = True
                    xb1 = int(float(line[2]))
                    xb2 = int(float(line[4]))
                    yb1 = int(float(line[3]))
                    yb2 = int(float(line[5]))
                    # Obtain the measures of depth from bounding box
                    bounding_depth = matrix_depth[yb1:yb2, xb1:xb2]
                    bounding_depth = np.reshape(bounding_depth, (1, np.size(bounding_depth)))
                    # self.show_bounding_depth(bounding_depth, YOLO_depth, self.labels[int(line[0])])
                    bounding_depth = self.filter_data_depth_before_median(bounding_depth)
                    measure_depth = statistics.median(bounding_depth)
                    # measure_depth = statistics.median(bounding_depth[0, :])

                for obj in range(self.n_classes):
                    # Case to detect an object with the object class
                    if int(line[0]) is obj:

                        if xcoord_object_detected[obj] is None:
                            # Create an empty list
                            xcoord_object_detected[obj] = []
                        # Write in the list the horizontal coordinate value
                        xcoord_object_detected[obj].append(xc)
                        if ycoord_object_detected[obj] is None:
                            # Create an empty list
                            ycoord_object_detected[obj] = []
                        # Write in the list the vertical coordinate value
                        ycoord_object_detected[obj].append(yc)

                        if depth:
                            if depth_object_detected[obj] is None:
                                # Create an empty list
                                depth_object_detected[obj] = []
                                # Write in the list the depth value
                                # depth_object_detected[obj].append(float(line[self.coord_depth]))
                                depth_object_detected[obj].append(measure_depth)
                            else:
                                # Write in the list the depth value
                                # depth_object_detected[obj].append(float(line[self.coord_depth]))
                                depth_object_detected[obj].append(measure_depth)
                    """
                    if xcoord_object_detected[obj] is not None:
                        # Sort the list from least to greatest
                        xcoord_object_detected[obj] = sorted(xcoord_object_detected[obj])
    
                    if ycoord_object_detected[obj] is not None:
                        # Sort the list from least to greatest
                        ycoord_object_detected[obj] = sorted(ycoord_object_detected[obj])
    
                    if depth and depth_object_detected[obj] is not None:
                        # Sort the list from least to greatest
                        depth_object_detected[obj] = sorted(depth_object_detected[obj])
                    
                    """

        return xcoord_object_detected, ycoord_object_detected, depth_object_detected

    ####################################################################################################
    # FUNCTION FOR READ THE DATA FROM FILE ENCODERS
    ####################################################################################################
    @staticmethod
    def read_data_encoder(file_k, file_kM1=None):
        # Case to read only one encoder data (first capture)
        if file_kM1 is None:
            X_kM1 = 0.0
            Y_kM1 = 0.0
            T_kM1 = 0.0
        # Case to read the actual and previous encoders data
        else:
            # Data from Encoder_kM1 (Previous Image --> k-1)
            encoder_kM1 = file_kM1.readline()
            encoder_kM1 = encoder_kM1.split('\n')[0]
            encoder_kM1 = encoder_kM1.split(',')
            # Obtain the coordinates from wheelchair (Actual State)
            X_kM1 = float(encoder_kM1[0].split('[')[1])
            Y_kM1 = float(encoder_kM1[1].split(' ')[1])
            T_kM1 = float(encoder_kM1[2].split(' ')[1].split(']')[0])

        # Data from Encoder_k (Actual Image --> k)
        encoder_k = file_k.readline()
        encoder_k = encoder_k.split('\n')[0]
        encoder_k = encoder_k.split(',')
        # Obtain the coordinates from wheelchair (Actual State)
        X_k = float(encoder_k[0].split('[')[1])
        Y_k = float(encoder_k[1].split(' ')[1])
        T_k = float(encoder_k[2].split(' ')[1].split(']')[0])

        return X_k - X_kM1, Y_k - Y_kM1, T_k - T_kM1

    ####################################################################################################
    # FUNCTION FOR RESTART A DICTIONARY
    ####################################################################################################
    @staticmethod
    def restart_dictionary(dictionary):
        # Loop for travel on each key
        for __, sub_dict in dictionary.items():
            values = list(sub_dict.values())
            if not bool(values[0]) and not bool(values[1]):
                # Empty List
                pass
            else:
                for key in sub_dict:
                    if key == 'u_clust':
                        break
                    else:
                        sub_dict[key] = []

    #####################################################################################################
    # FUNCTION FOR REPRESENT THE RESULTS OF ESTIMATION POSE FROM OBJECTS
    #####################################################################################################

    def graphics_representation(self, list_of_results, name_sequence, path2save):
        # Dictionary with the colors for each class of object
        """
        dict_colors = {
            'window': "c",
            'door': "orange",
            'elevator': "pink",
            'fireext': "r",
            'plant': "g",
            'bench': "grey",
            'firehose': "darkred",
            'lightbox': "darkblue",
            'column': "m",
            'toilet': "y",
            'person': "k",
            'fallen': "brown"
        }
        """
        # Directory for save the images results
        directory_results = './Results/ResultsTracking/'+path2save+'/'

        dict_colors = {
            'window': "c",
            'door': "magenta",
            'elevator': "pink",
            'fireext': "r",
            'plant': "gold",
            'bench': "grey",
            'firehose': "darkred",
            'lightbox': "darkblue",
            'column': "limegreen"
        }
        # Auxiliar vars for save the max value of each list objects
        maximum_y = 0.0
        maximum_x = 0.0
        parameters = {'xtick.labelsize': 25,
                      'ytick.labelsize': 25}
        plt.rcParams.update(parameters)
        # Loop for travel the labels
        for key in self.labels:
            # Loop for travel each dictionary from results list
            try:
                for index_list in list_of_results:
                    x = index_list[key]['d_trans']
                    y = index_list[key]['d_from_ref1']
                    if not bool(x) and not bool(y):
                        pass
                    else:
                        m_y = max(y)
                        m_x = max(x)
                        if m_y > maximum_y:
                            maximum_y = m_y
                        if m_x > maximum_x:
                            maximum_x = m_x
                        plt.scatter(x, y, s=200, c=dict_colors[key])
            except:
                x = list_of_results[key]['d_trans']
                y = list_of_results[key]['d_from_ref1']
                if not bool(x) and not bool(y):
                    pass
                else:
                    m_y = max(y)
                    m_x = max(x)
                    if m_y > maximum_y:
                        maximum_y = m_y
                    if m_x > maximum_x:
                        maximum_x = m_x
                    plt.scatter(x, y, s=200, c=dict_colors[key])

        plt.xticks(np.arange(int(-maximum_x-1), int(maximum_x+1), step=1))
        plt.yticks(np.arange(0.0, maximum_y, step=3))
        plt.savefig(directory_results+name_sequence+'.png')
        plt.cla()

    ####################################################################################################
    # FUNCTION FOR ESTIMATE THE POSE
    ####################################################################################################

    def estimation_pose_object(self, x_coord, depth, inc_y, inc_x, inc_theta):
        """
        Function to calculate the estimation pose of the object using the intersection between two
        lines.

        :param x_coord: coordinate 'x' of the object in the image
        :param depth: parameter of depth from object in meters
        :param inc_y: current displacement with the wheelchair in y in cm
        :return: the intersection between the line and circumference
        """
        """
        # Angle between object and robot's reference system
        angle_p = (self.aperture_cam / self.width) * (x_coord - self.width / 2)
        y_p = depth * 100
        x_p = (depth * 100) * math.tan(angle_p)
        # Components between object and reference system of init node movement
        R_x = (inc_x + x_p * math.cos(inc_theta) - y_p * math.sin(inc_theta))
        R_y = (inc_y + x_p * math.sin(inc_theta) + y_p * math.cos(inc_theta))

        """
        # Angle's detection of the object in the image
        angle = (self.aperture_cam / self.width) * (x_coord - self.width / 2)

        # Calculate the point of circumference using the depth data (radio) and angle
        R_y = (depth * 100) + inc_y
        R_x = (depth * 100) * math.tan(angle)

        return R_x, R_y

    ####################################################################################################################
    # FUNCTION FOR PROCESS OF CLUSTERING
    ####################################################################################################################
    def clustering_objects(self, list_of_results):
        """
        for objectClass in self.labels:
            List_dist_ref1 = []
            List_dist_trans = []

            for dictionary in list_of_results:
                d_from_ref1 = dictionary[objectClass]['d_from_ref1']
                d_trans = dictionary[objectClass]['d_trans']

                if not d_from_ref1:
                    continue
                elif len(d_from_ref1) == 1:
                    List_dist_ref1.append(d_from_ref1[-1])
                    List_dist_trans.append(d_trans[-1])
                else:
                    List_dist_ref1.extend(d_from_ref1)
                    List_dist_trans.extend(d_trans)

            while len(List_dist_ref1) > 1:
                distances = np.sqrt((np.subtract.outer(List_dist_ref1, List_dist_ref1) ** 2) + (
                            np.subtract.outer(List_dist_trans, List_dist_trans) ** 2))

                # Ignore diagonal elements (which are zero) and upper triangle of matrix
                distances[np.triu_indices(len(distances))] = np.inf

                min_index = np.argmin(distances)
                min_row, min_col = np.unravel_index(min_index, distances.shape)

                if distances[min_row, min_col] < self.objects_estimation[objectClass]['u_clust'] ** 2:
                    List_dist_ref1[min_row] = 0.5 * (List_dist_ref1[min_row] + List_dist_ref1[min_col])
                    List_dist_trans[min_row] = 0.5 * (List_dist_trans[min_row] + List_dist_trans[min_col])
                    List_dist_ref1.pop(min_col)
                    List_dist_trans.pop(min_col)
                else:
                    break

            self.final_results_clustering[objectClass]['d_from_ref1'] = List_dist_ref1
            self.final_results_clustering[objectClass]['d_trans'] = List_dist_trans

        """
        # Loop for travel each object class
        for objectClass in self.labels:
            # Local variables for clustering
            List_dist_ref1 = []
            List_dist_trans = []
            # Loop for travel each dictionary form list of results estimations
            for dictionary in list_of_results:
                # Comprobate if the list is empty, have one value or more than one
                if not bool(dictionary[objectClass]['d_from_ref1']):
                    pass
                elif len(dictionary[objectClass]['d_from_ref1']) == 1:
                    List_dist_ref1.append(dictionary[objectClass]['d_from_ref1'][-1])
                    List_dist_trans.append(dictionary[objectClass]['d_trans'][-1])
                else:
                    for i in range(len(dictionary[objectClass]['d_from_ref1'])):
                        List_dist_ref1.append(dictionary[objectClass]['d_from_ref1'][i])
                        List_dist_trans.append(dictionary[objectClass]['d_trans'][i])

            while len(List_dist_ref1) > 1:
                # Se crea una matriz de dimensiones nxn
                dd = 1000 * np.ones([len(List_dist_ref1), len(List_dist_ref1)])
                # Tomamos el primer vector(podria ser uno al azar)
                # y calculamos el resto de las
                # distancias a todos los demas vectores.
                for n in range(len(List_dist_ref1)):
                    for m in range(n + 1, len(List_dist_ref1), 1):
                        dd[n, m] = ((List_dist_ref1[n] - List_dist_ref1[m]) ** 2 +
                                    (List_dist_trans[n] - List_dist_trans[m]) ** 2)

                row, col = np.where(dd == np.min(dd))

                if dd[row[0], col[0]] < self.objects_estimation[objectClass]['u_clust'] ** 2:
                    List_dist_ref1[row[0]] = 0.5 * (List_dist_ref1[row[0]] + List_dist_ref1[col[0]])
                    List_dist_trans[row[0]] = 0.5 * (List_dist_trans[row[0]] + List_dist_trans[col[0]])
                    List_dist_ref1.pop(col[0])
                    List_dist_trans.pop(col[0])
                else:
                    break

            self.final_results_clustering[objectClass]['d_from_ref1'] = List_dist_ref1
            self.final_results_clustering[objectClass]['d_trans'] = List_dist_trans

    ####################################################################################################################
    # PRINT THE RESULTS FROM TRACKING
    ####################################################################################################################

    def results_tracking(self, sequence, y2, path2save):

        directory_results = './Results/ResultsTracking/'+path2save+'/'
        # First: comprobate if the directory exist or not. In case to not exist, must be created
        if os.path.exists(directory_results):
            pass
        else:
            # Create the directories
            os.makedirs(directory_results, exist_ok=True)

        # Open the file
        fid = open(directory_results+'%s_Results.txt' % sequence, 'w')
        file_sequence = sequence.split("/")
        # fid = open(directory_results + '%s_Results.txt' % file_sequence[1], 'w')

        # Distance of Nodes
        fid.write("Distance nodes: %0.2f\n" % (y2 / 100))

        for n in self.labels:
            print('----------------------------------------------------')
            print('Objects_tracking')
            print('Class: %s' % n)

            if self.final_results_clustering[n] is not None:
                for i in range(len(self.final_results_clustering[n]['d_from_ref1'])):
                    print("Object: %d" % (i + 1))
                    df = self.final_results_clustering[n]['d_from_ref1'][i]
                    dt = self.final_results_clustering[n]['d_trans'][i]
                    # Generate a List with the poses of objects from same class
                    print('-------------------------------------------')
                    print("estimation_obj_dist_ref_node: %0.4f (meters)" % df)
                    print("estimation_transversal_distance: %0.4f (meters)" % dt)
                    print('-------------------------------------------')
                    fid.write("Class: %s , Distance (meters): %0.2f , Transversal(m): %0.2f\n" % (n, df, dt))

        # Close the file when finnish the writing task
        fid.close()

    ####################################################################################################################
    # FUNCTION FOR VISUALIZE IN THE IMAGES THE PROCESS OF TRACKING OBJECTS
    ####################################################################################################################

    def visualize_tracking(self, sequence, first_img, last_img):

        directory_tracking_img = './ResultsTracking/ImgResultsTrack/'+sequence+'/'
        # First: comprobate if the directory exist or not. In case to not exist, must be created
        if os.path.exists(directory_tracking_img):
            pass
        else:
            # Create the directories
            os.makedirs(directory_tracking_img, exist_ok=True)

        # Path where is located the images
        path = './' + sequence + '/imgTestResults/ImgColor%d.png'

        for index in range(first_img + 1, last_img + 1):
            file_img = (path % index)
            img = cv2.imread(file_img)
            head_tail = os.path.split(file_img)
            filename_img_represent = head_tail[1]  # filename without path
            for n in range(self.n_classes):
                # print(objects_tracking[n])
                if self.objects_tracking[n] is not None:
                    for i in range(len(self.objects_tracking[n])):
                        d = self.objects_tracking[n][i]
                        # check if the tracked object includes the image
                        if file_img in d['Images']:
                            # If the tracked object includes the image, determine the position of the image
                            # in the tracked positions.

                            ind_img_object = d['Images'].index(file_img)
                            for j in range(ind_img_object):
                                x1 = d['x_coords'][j]
                                y1 = d['y_coords'][j]
                                x2 = d['x_coords'][j + 1]
                                y2 = d['y_coords'][j + 1]
                                line_thickness = 4
                                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=line_thickness)
                                img = cv2.circle(img, (x1, y1), radius=6, color=(255, 0, 0), thickness=-1)
                                img = cv2.circle(img, (x2, y2), radius=6, color=(255, 0, 0), thickness=-1)

            # Write the image in the directory
            cv2.imwrite(directory_tracking_img + filename_img_represent, img)

    ####################################################################################################################
    # MAIN FUNCTION FOR EVALUATING A SEQUENCE OF IMAGES
    ####################################################################################################################

    def tracking_sequence(self, sequence, first_image=None, last_image=None, path2save=None):
        """
        This function evaluate the tracking objects for a sequence of images
        :param sequence: the path where is saved the sequence of images
        In case to evaluate a part of sequence this inputs delimit process of tracking
        :param first_image: first image of the sequence
        :param last_image: last image of the sequence
        :return:
        """
        # Path for detect de sequence of images, the files from encoder and file for depth
        path_imgs = './Resources/Tracking/'+sequence+'/imgTestResults'
        path_enco = './Resources/Tracking/'+sequence+'/Encoders'
        path_depth = './Resources/Tracking/'+sequence+'/DataDepth'

        # Delimit the bounds for sequence
        if first_image is None and last_image is None:
            # In case that not detect any bounds for sequence, the method run the process for all images of sequence
            first_image = 0
            last_image = int(len(os.listdir(path_imgs)) / 2 - 1)

        # Loop to go through the sequence of images comparing pairs of images (the current one and the previous one)

        index = first_image
        hist_Delta_y = 0.0
        hist_Delta_x = 0.0
        hist_Delta_Theta = 0.0
        self.final_results_clustering = dict((key, {'d_from_ref1': [], 'd_trans': []}) for key in self.labels)

        while index <= last_image:
            # File of Detections objects (First Image)
            file_img = path_imgs + '/ImgColor%d.png' % index             # File image with extension .png
            file_detection = path_imgs + '/ImgColor%d.txt' % index       # File of detections with extension .txt
            file_depth = path_depth + '/Img%d.txt' % index               # File with the depth info with extension .txt

            # File of Detections objects (First Image) CLASIFICACION DE NODOS
            ####################################################################################################
            # Integration of IbPRIA22 Nodes
            print("Node-classification (IbPRIA22)")
            path_lidar = path_imgs.replace('imgTestResults', 'Capture_Lidar')
            file_lidar = os.path.join(path_lidar, 'Lidar%d.npy' % index)  # File image with extension .pny
            nbins = 50  # number of intervals of the node signature
            dist_normalization = 12
            vector_node = node_classification.node_classification(file_lidar, nbins, flag_signature=False)
            # for n in range(nbins):
            #     print("n:%d,%0.4g" %(n,vector_node[n]/dist_normalization))

            # SVM Hyper-parameters
            C = 16
            g = 1
            file_vector = "vector_prueba.txt"
            fid_file_vector = open(file_vector, 'w')
            fid_file_vector.writelines("0")
            for n in range(nbins):
                fid_file_vector.writelines(" %d:%0.3g" % (n, vector_node[n] / dist_normalization))
            fid_file_vector.close()

            file_model = "./Resources/nodes%d_analog_train.txt.model" % (nbins)
            if not (os.path.exists(file_model)):
                print("File svm-model not found: %s" % (file_model))

            file_prediction = "./prueba.txt.predict"
            if not (os.path.exists(file_prediction)):
                fid_file_prediction = open(file_prediction, 'w')
                fid_file_prediction.close()
            # Para Linux
            # os.system("./Resources/libsvm-3.25/svm-predict %s %s %s" % (file_vector, file_model, file_prediction))
            # Para windows
            # path_svm = os.path.join("C:/Users/die_d/repositorios/tracking-eval-weights/Resources/libsvm-3.25/")
            # os.system(path_svm + "svm-predict %s %s %s" % (file_vector, file_model, file_prediction))

            nodes = ["NoNode", "EndNode", "NodeT", "CrossNode", "NodeL", "OpenNode"]

            # with open(file_prediction) as fid1:
            #    line = fid1.readline()
            #    label = int(line[0])
            #    print(label),print(nodes[label])
            #    print("node_classification:%s\n" % nodes[label])

            os.remove(file_vector)
            os.remove(file_prediction)

            ####################################################################################################

            # Read the coordinates of the objects from YOLO
            xc_objects, yc_objects, depth_objects = self.read_CoordYOLO(file_detection, file_depth)
            # Comprobate that the distance is under of umbral from data depth
            for xc_data, yc_data, depth_data in zip(xc_objects, yc_objects, depth_objects):
                if depth_data is None:
                    pass
                elif len(depth_data) == 1:
                    # The object detected is considerate more away than the umbral camera, so the detection
                    # is discarded
                    if depth_data[-1] > 10.0 or depth_data[-1] < 1.0:
                        depth_data.pop(-1)
                        xc_data.pop(-1)
                        yc_data.pop(-1)
                else:
                    # Case of list with more of one value
                    ind = 0
                    while ind < len(depth_data):
                        if (depth_data[ind] > 10.0 or depth_data[ind] < 1.0) and len(depth_data) != 1:
                            depth_data.pop(ind)
                            xc_data.pop(ind)
                            yc_data.pop(ind)
                            ind = 0
                        elif (depth_data[ind] > 10.0 or depth_data[ind] < 1.0) and len(depth_data) == 1:
                            depth_data.pop(-1)
                            xc_data.pop(-1)
                            yc_data.pop(-1)
                            break
                        else:
                            ind += 1
            self.dict_YOLO_Camera['xc_objects'] = xc_objects
            self.dict_YOLO_Camera['yc_objects'] = yc_objects
            self.dict_YOLO_Camera['depth_objects'] = depth_objects

            self.list_YOLO_Camera_results.append(copy.deepcopy(self.dict_YOLO_Camera))

            self.dict_YOLO_Camera['xc_objects'] = None
            self.dict_YOLO_Camera['yc_objects'] = None
            self.dict_YOLO_Camera['depth_objects'] = None

            # File of Encoders from wheelchair
            if index == 0:
                file_encoder_k = open((path_enco + '/Encoders%d.txt' % index), 'r')
                Delta_x, Delta_y, Delta_Theta = self.read_data_encoder(file_encoder_k)
                hist_Delta_y += Delta_y
                hist_Delta_x += Delta_x
                hist_Delta_Theta += Delta_Theta
            else:
                file_encoder_k = open((path_enco + '/Encoders%d.txt' % index), 'r')
                file_encoder_kM1 = open((path_enco + '/Encoders%d.txt' % (index - 1)), 'r')
                Delta_x, Delta_y, Delta_Theta = self.read_data_encoder(file_encoder_k, file_encoder_kM1)
                hist_Delta_y += Delta_y
                hist_Delta_x += Delta_x
                hist_Delta_Theta += Delta_Theta

            print('\n#############################################')
            print(file_img)
            print("inc_x:%0.3g,inc_y:%0.3g,inc_t:%0.3g" % (Delta_x, Delta_y, Delta_Theta))
            print('-------------------------')
            print("Coordinates of centroid's objects (x,y) from " + 'ImgColor%d.png' % index)
            print(xc_objects)
            print(yc_objects)
            print('-------------------------')

            ########################################################################################
            # ESTIMATION POSE OBJECTS FOR EACH IMAGE OF SEQUENCE
            ########################################################################################
            for list_object, list_depth, class_object in zip(xc_objects, depth_objects, self.labels):
                if (list_object and list_depth) is None:
                    pass
                else:
                    for index_object, index_depth in zip(list_object, list_depth):
                        d_trans_estimate, d_ref1_estimate = self.estimation_pose_object(index_object,
                                                                                        index_depth,
                                                                                        hist_Delta_y,
                                                                                        hist_Delta_x,
                                                                                        hist_Delta_Theta)
                        self.objects_estimation[class_object]['d_from_ref1'].append(round(d_ref1_estimate/100, 2))
                        self.objects_estimation[class_object]['d_trans'].append(round(d_trans_estimate/100, 2))

            # Save the results of objects estimation on a list of results
            self.list_results.append(copy.deepcopy(self.objects_estimation))
            # self.graphics_representation(self.list_results)
            # Restart the dictionary objects_estimation for next iteration
            self.restart_dictionary(self.objects_estimation)

            # Increment the index
            index = index + 1

        # PLOT THE RESULTS
        self.graphics_representation(self.list_results, sequence, path2save)

        # CLUSTERING THE OBJECTS
        self.clustering_objects(self.list_results)

        self.graphics_representation(self.final_results_clustering, sequence, path2save)

        # These sentences go out from while loop
        # Process for obtain the results from tracking
        self.results_tracking(sequence, hist_Delta_y, path2save)
        # Visualize images with tracked objects
        # Represent objects in the second image of each pair of consecutive frames
        # self.visualize_tracking(sequence, first_image, last_image)

        # Here return the results from encoders for distance nodes
        # Remember that hist_Delta_y is in cm
        return round(hist_Delta_y / 100.0, 2)

########################################################################################################################
# END CLASS DETECTION
########################################################################################################################
