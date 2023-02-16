import numpy as np
import os
from re import findall
from copy import deepcopy


class Map(object):
    # Builder of class map
    def __init__(self):
        # Create the object from class Edge where is saved the annotations objects
        self.EDGE_AN = Edge()
        self.NODE_AN = Nodes()
        self.NODE_AN.read_nodes_txt('P_edge.txt')
        self.NODE_AN.create_nodes_edge()
        self.EDGE_AN.read_objects_edge_txt()
        # Complete the key 'objects' from variable LIST_NODES_EDGES from class 'Nodes'
        index = 0
        for EDGE in self.NODE_AN.LIST_NODES_EDGES:
            EDGE['objects'] = self.EDGE_AN.LIST_OBJECTS_EDGES[index]
            index = index + 1


class Edge(object):
    # Builder of class edge
    def __init__(self):
        # Path where is saved the data
        self.path = './Resources/'
        self.path_annotated_d_from_ref = './Resources/EvalWeights/Edges_d_from_ref1/'
        self.path_annotated_d_trans = './Resources/EvalWeights/Transversal_dist/'
        # List with objects classes
        self.class_object = ['window', 'door', 'elevator',
                             'fireext', 'plant', 'bench',
                             'firehose', 'lightbox', 'column']
        # Charge the matrix of Edges and create an empty list of EDGES
        Matrix_Edges = np.loadtxt('./Resources/EvalWeights/Matrix_Edges.txt')
        self.LIST_OBJECTS_EDGES = [None] * int(Matrix_Edges.max())
        # The loop is necessary for create different dictionaries that will have different referencies, with this action
        # avoid that one change on one key don´t affect the other keys of dictionaries in different position in the list
        for i in range(int(Matrix_Edges.max())):
            self.LIST_OBJECTS_EDGES[i] = {
                                            'window': {'d_from_ref1': [], 'd_trans': []},
                                            'door': {'d_from_ref1': [], 'd_trans': []},
                                            'elevator': {'d_from_ref1': [], 'd_trans': []},
                                            'fireext': {'d_from_ref1': [], 'd_trans': []},
                                            'plant': {'d_from_ref1': [], 'd_trans': []},
                                            'bench': {'d_from_ref1': [], 'd_trans': []},
                                            'firehose': {'d_from_ref1': [], 'd_trans': []},
                                            'lightbox': {'d_from_ref1': [], 'd_trans': []},
                                            'column': {'d_from_ref1': [], 'd_trans': []}
                                        }

    def read_objects_edge_txt(self):
        # Read each file txt
        for file_ID in os.listdir(self.path_annotated_d_from_ref):
            # FIRST STEP: Load the txt where is saved the frontal distance from objects
            file_d_from_ref = np.loadtxt(self.path_annotated_d_from_ref+file_ID)
            # Obtain the index for the List Edges from name of file .txt
            index_list = [float(s) for s in findall(r'-?\d+\.?\d*', file_ID)]
            # Save the frontal distance objects in each Edge
            index_object = 0
            for Obj in self.class_object:
                obj_dist_ref1 = file_d_from_ref[index_object]
                for j in range(len(obj_dist_ref1)):
                    if obj_dist_ref1[j] != 0:
                        self.LIST_OBJECTS_EDGES[int(index_list[-1])-1][Obj]['d_from_ref1'].append(abs(obj_dist_ref1[j]))
                # Increment the index for next iteration
                index_object = index_object + 1

        # Loop for read the files txt with the transversal distance from objects
        for file_ID in os.listdir(self.path_annotated_d_trans):
            # FIRST STEP: Load the txt where is saved the frontal distance from objects
            file_d_trans = np.loadtxt(self.path_annotated_d_trans + file_ID)
            # Obtain the index for the List Edges from name of file .txt
            index_list = [float(s) for s in findall(r'-?\d+\.?\d*', file_ID)]
            # Save the transversal distance objects in each Edge
            index_object = 0
            for Obj in self.class_object:
                obj_dist_trans = file_d_trans[index_object]
                for j in range(len(obj_dist_trans)):
                    if obj_dist_trans[j] != 0:
                        # Save the dictionary
                        self.LIST_OBJECTS_EDGES[int(index_list[-1])-1][Obj]['d_trans'].append(obj_dist_trans[j])
                # Increment the index for next iteration
                index_object = index_object + 1

########################################################################################################################


class Nodes(object):
    def __init__(self):
        self.LIST_NODES = {
            1: {'class': 'E', 'pose': [1850, 3283], 'Node': [2], 'Ang': [0]},
            2: {'class': 'T', 'pose': [1576, 3283], 'Node': [1, 3, 63], 'Ang': [0, -91, 92]},
            3: {'class': 'T', 'pose': [1567, 2832], 'Node': [2, 4, 5], 'Ang': [0, 91, -178]},
            4: {'class': 'E', 'pose': [702, 2836], 'Node': [3], 'Ang': [0]},
            5: {'class': 'T', 'pose': [1576, 2488], 'Node': [3, 6, 7], 'Ang': [0, 89, 177]},
            6: {'class': 'E', 'pose': [1346, 2488], 'Node': [5], 'Ang': [0]},
            7: {'class': 'T', 'pose': [1563, 2139], 'Node': [5, 8, 9], 'Ang': [0, 92, -180]},
            8: {'class': 'E', 'pose': [707, 2139], 'Node': [7], 'Ang': [0]},
            9: {'class': 'L', 'pose': [1555, 1951], 'Node': [7, 10], 'Ang': [0, -133]},
            10: {'class': 'L', 'pose': [1912, 1594], 'Node': [9, 11], 'Ang': [0, -142]},
            11: {'class': 'T', 'pose': [2117, 1569], 'Node': [10, 12, 13], 'Ang': [0, 97, -170]},
            12: {'class': 'E', 'pose': [2117, 713], 'Node': [11], 'Ang': [0]},
            13: {'class': 'T', 'pose': [2461, 1590], 'Node': [11, 14, 15], 'Ang': [0, 87, 177]},
            14: {'class': 'E', 'pose': [2461, 1369], 'Node': [13], 'Ang': [0]},
            15: {'class': 'T', 'pose': [2801, 1590], 'Node': [13, 16, 17], 'Ang': [0, 90, 179]},
            16: {'class': 'E', 'pose': [2801, 729], 'Node': [15], 'Ang': [0]},
            17: {'class': 'T', 'pose': [3252, 1582], 'Node': [15, 18, 19], 'Ang': [0, -89, -176]},
            18: {'class': 'E', 'pose': [3252, 1881], 'Node': [17], 'Ang': [0]},
            19: {'class': 'T', 'pose': [3697, 1604], 'Node': [17, 20, 21], 'Ang': [0, 87, 172]},
            20: {'class': 'E', 'pose': [3691, 745], 'Node': [19], 'Ang': [0]},
            21: {'class': 'T', 'pose': [4037, 1574], 'Node': [19, 22, 23], 'Ang': [0, 95, -175]},
            22: {'class': 'E', 'pose': [4037, 1383], 'Node': [21], 'Ang': [0]},
            23: {'class': 'T', 'pose': [4383, 1574], 'Node': [21, 24, 25], 'Ang': [0, 90, -174]},
            24: {'class': 'E', 'pose': [4383, 727], 'Node': [23], 'Ang': [0]},
            25: {'class': 'L', 'pose': [4609, 1598], 'Node': [23, 26], 'Ang': [0, -140]},
            26: {'class': 'L', 'pose': [4949, 1944], 'Node': [25, 27], 'Ang': [0, -134]},
            27: {'class': 'T', 'pose': [4943, 2152], 'Node': [26, 28, 29], 'Ang': [0, 88, 179]},
            28: {'class': 'E', 'pose': [5784, 2147], 'Node': [27], 'Ang': [0]},
            29: {'class': 'T', 'pose': [4937, 2492], 'Node': [27, 30, 31], 'Ang': [0, 91, 178]},
            30: {'class': 'E', 'pose': [5140, 2498], 'Node': [29], 'Ang': [0]},
            31: {'class': 'T', 'pose': [4943, 2838], 'Node': [29, 32, 33], 'Ang': [0, 91, -177]},
            32: {'class': 'E', 'pose': [5796, 2832], 'Node': [31], 'Ang': [0]},
            33: {'class': 'T', 'pose': [4925, 3279], 'Node': [31, 34, 35], 'Ang': [0, -93, 179]},
            34: {'class': 'E', 'pose': [4651, 3285], 'Node': [33], 'Ang': [0]},
            35: {'class': 'T', 'pose': [4919, 3736], 'Node': [33, 36, 37], 'Ang': [0, 89, 172]},
            36: {'class': 'E', 'pose': [5784, 3730], 'Node': [35], 'Ang': [0]},
            37: {'class': 'T', 'pose': [4961, 4076], 'Node': [35, 38, 39], 'Ang': [0, 97, -170]},
            38: {'class': 'E', 'pose': [5152, 4076], 'Node': [37], 'Ang': [0]},
            39: {'class': 'T', 'pose': [4943, 4427], 'Node': [37, 40, 41], 'Ang': [0, 87, -178]},
            40: {'class': 'E', 'pose': [5790, 4427], 'Node': [39], 'Ang': [0]},
            41: {'class': 'L', 'pose': [4925, 4636], 'Node': [39, 42], 'Ang': [0, -143]},
            42: {'class': 'L', 'pose': [4597, 5000], 'Node': [41, 43], 'Ang': [0, -135]},
            43: {'class': 'T', 'pose': [4377, 5012], 'Node': [42, 44, 45], 'Ang': [0, 93, -175]},
            44: {'class': 'E', 'pose': [4377, 5840], 'Node': [43], 'Ang': [0]},
            45: {'class': 'T', 'pose': [4043, 5000], 'Node': [43, 46, 47], 'Ang': [0, 90, 179]},
            46: {'class': 'E', 'pose': [4037, 5185], 'Node': [45], 'Ang': [0]},
            47: {'class': 'T', 'pose': [3697, 4994], 'Node': [45, 48, 49], 'Ang': [0, 89, 179]},
            48: {'class': 'E', 'pose': [3697, 5846], 'Node': [47], 'Ang': [0]},
            49: {'class': 'T', 'pose': [3250, 4994], 'Node': [47, 50, 51], 'Ang': [0, -89, -177]},
            50: {'class': 'E', 'pose': [3256, 4696], 'Node': [49], 'Ang': [0]},
            51: {'class': 'T', 'pose': [2805, 4974], 'Node': [49, 52, 53], 'Ang': [0, 87, 173]},
            52: {'class': 'E', 'pose': [2805, 5829], 'Node': [51], 'Ang': [0]},
            53: {'class': 'T', 'pose': [2454, 4998], 'Node': [51, 54, 55], 'Ang': [0, 94, -175]},
            54: {'class': 'E', 'pose': [2454, 5179], 'Node': [53], 'Ang': [0]},
            55: {'class': 'T', 'pose': [2114, 4992], 'Node': [53, 56, 57], 'Ang': [0, 89, 179]},
            56: {'class': 'E', 'pose': [2114, 5852], 'Node': [55], 'Ang': [0]},
            57: {'class': 'L', 'pose': [1915, 4992], 'Node': [55, 58], 'Ang': [0, -132]},
            58: {'class': 'L', 'pose': [1564, 4605], 'Node': [57, 59], 'Ang': [0, -140]},
            59: {'class': 'T', 'pose': [1558, 4424], 'Node': [58, 60, 61], 'Ang': [0, 92, -177]},
            60: {'class': 'E', 'pose': [698, 4424], 'Node': [59], 'Ang': [0]},
            61: {'class': 'T', 'pose': [1564, 4090], 'Node': [59, 62, 63], 'Ang': [0, 89, 179]},
            62: {'class': 'E', 'pose': [1342, 4090], 'Node': [61], 'Ang': [0]},
            63: {'class': 'T', 'pose': [1564, 3733], 'Node': [2, 61, 64], 'Ang': [0, 178, -92]},
            64: {'class': 'E', 'pose': [709, 3739], 'Node': [63], 'Ang': [0]}
        }

        # Path where is saved the data
        self.path = './Resources/EvalWeights/'

        # Attributes of this class (Part of nodes)
        self.init_node = []
        self.dest_node = []
        self.dist = []
        self.data_nodes = []

        # Charge the matrix of Edges and create an empty list of EDGES
        self.Matrix_Edges = np.loadtxt('./Resources/EvalWeights/Matrix_Edges.txt')
        # Create the list where will be saved de info about nodes annotation
        self.LIST_NODES_EDGES = [None] * int(self.Matrix_Edges.max())

    def read_nodes_txt(self, file_txt):
        # FIRST: open the file txt
        file = open(self.path+file_txt, 'r')

        # Loop for read line to line
        for line in file:
            data_split = line.split(' ')
            try:
                self.init_node.append(int(data_split[0]))
                self.dest_node.append(int(data_split[1]))
                self.dist.append(float(data_split[2].split('\n')[0]))
            except ValueError:
                pass

    def create_nodes_edge(self):
        # FIRST STEP: loop for save the data
        for i in range(self.Matrix_Edges.shape[0]):
            for j in range(self.Matrix_Edges.shape[1]):
                if self.Matrix_Edges[i][j] != 0:
                    # Obtain the distance between the nodes
                    index = 0
                    for k in range(len(self.init_node)):
                        if (i+1) == self.init_node[k] and (j+1) == self.dest_node[k]:
                            break
                        elif (i+1) == self.dest_node[k] and (j+1) == self.init_node[k]:
                            break
                        index = index + 1
                    # Possible connection
                    self.LIST_NODES_EDGES[int(self.Matrix_Edges[i][j])-1] = {'Ref1': i + 1,
                                                                             'Ref2': j + 1,
                                                                             'dist': self.dist[index],
                                                                             'objects': None}

########################################################################################################################
# END CLASS
########################################################################################################################
