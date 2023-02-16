# from manual_tracking import ManualTrack
import copy
from Tracking_Objects_v4 import Detection
from estimate_weights_v10_64Nodos import EstimationWeights
from map_annotation import Map
# from detections import dict_detections_nodes
import numpy as np
import sys
import json
import os

########################################################################################################################
# CAPTURA DE LOS ARGUMENTOS DE ENTRADA
########################################################################################################################

print("Procesando fichero " + sys.argv[1] + " ...")
with open(os.path.join("Resources", sys.argv[1])) as file_json:
    data_diccionario = json.load(file_json)

# Procesamiento de la secuencia
for departamento in data_diccionario:
    # Comprobar si esta vacio o no la lista en caso de no estarlo devuelve un True
    if bool(data_diccionario[departamento]):
        # Bucle para crear la secuencia
        sequences = []
        dict_detections_nodes = []
        for edge_detected in data_diccionario[departamento]:
            list_aux = edge_detected.split(" ")
            sequences.append([list_aux[0], float(list_aux[1])])
            dict_detections_nodes.append({'Node_init': list_aux[2],
                                          'Node_dest': list_aux[3],
                                          'dist': float(list_aux[4])})

########################################################################################################################
# PARTE DEL TRACKING
########################################################################################################################
# Object for detection (Tracking)
DET = Detection("classes.names", "YOLO_v1")

# Write the sequences to evaluate the tracking

# sequences = [["Secuencia1", 0.0], ["Secuencia2", 90.0], ["Secuencia3", 90.0], ["Secuencia4", 180.0],
#             ["Secuencia5", -90.0], ["Secuencia6", 0.0], ["Secuencia7", 90.0]]


# sequences = [["DepSur/Sec_N20N19", 0.0], ["DepSur/Sec_N19N21", 90.0], ["DepSur/Sec_N21N22", 90.0]]
# sequences = [["DepNorte/Sec_N52N51", 0.0], ["DepNorte/Sec_N51N53", 90.0], ["DepNorte/Sec_N53N54", 90.0]]
# sequences = [["DepEste/Sec_N36N35", 0.0], ["DepEste/Sec_N35N37", 90.0], ["DepEste/Sec_N37N38", 90.0]]
# sequences = [["DepOeste/Sec_N12N11", 0.0], ["DepOeste/Sec_N11N13", 90.0], ["DepOeste/Sec_N13N14", 90.0]]

"""
sequences = [["Secuencia1", 0.0], ["Secuencia2", 90.0], ["Secuencia5", -180.0],
             ["Secuencia6", 0.0], ["Secuencia7", 90.0]]
"""
# sequences = [["Secuencia2", 0.0]]

########################################################################################################################
# PARTE DE LA 'ANNOTATION'
########################################################################################################################
# Create the object from class Estimation
EW = EstimationWeights("classes.names")
# Create the annotations EDGES
MAP = Map()

########################################################################################################################
# PARTE DE LA 'DETECTION'
########################################################################################################################
# Tracking and Evaluating for three sequences
for sequence, id_seq in zip(sequences, range(len(sequences))):

    if (id_seq+1) == 1:
        print(sequence)
        DET.tracking_sequence(sequence[0])
        # Reestart some variable for next iteration
        DET.list_results = []
        DET.list_YOLO_Camera_results = []
        # Object Detection Sequence
        # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
        object_detections = DET.final_results_clustering
        node_detection = dict_detections_nodes[id_seq]
        # Eval the weight
        EW.evaluate_location(1,
                             MAP.NODE_AN.LIST_NODES,
                             MAP.NODE_AN.LIST_NODES_EDGES,
                             node_detection,
                             object_detections,
                             sequence[1])

    elif (id_seq+1) == 2:
        print(sequence)
        DET.tracking_sequence(sequence[0])
        # Reestart some variable for next iteration
        DET.list_results = []
        DET.list_YOLO_Camera_results = []
        # Object Detection Sequence
        # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
        object_detections = DET.final_results_clustering
        node_detection = dict_detections_nodes[id_seq]
        # Eval the weight
        EW.evaluate_location(2,
                             MAP.NODE_AN.LIST_NODES,
                             MAP.NODE_AN.LIST_NODES_EDGES,
                             node_detection,
                             object_detections,
                             sequence[1])

    elif (id_seq + 1) == 3:
        print(sequence)
        DET.tracking_sequence(sequence[0])
        # Reestart some variable for next iteration
        DET.list_results = []
        DET.list_YOLO_Camera_results = []
        # Object Detection Sequence
        # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
        object_detections = DET.final_results_clustering
        node_detection = dict_detections_nodes[id_seq]
        # Eval the weight
        EW.evaluate_location(3,
                             MAP.NODE_AN.LIST_NODES,
                             MAP.NODE_AN.LIST_NODES_EDGES,
                             node_detection,
                             object_detections,
                             sequence[1])

    if (id_seq + 1) > 3:
        # In case to explore more than 3 edges
        # Now the matrix_W1 will be the matrix_W2 from before iteration and the matrix W2 will be the matrix W3 from
        # before iteration
        EW.W_1 = copy.deepcopy(EW.W_2)
        EW.W_2 = copy.deepcopy(EW.W_3)
        # Now is calculated the new matrix W_12
        # First restart the matrix W_12
        EW.W_12 = np.zeros([64, 64, 64])
        # In this case is necessary evaluate the combinations for two Edges
        for i in range(EW.W_1.shape[0]):
            for j in range(EW.W_1.shape[1]):
                for k in range(EW.W_1.shape[1]):
                    EW.W_12[i][j][k] = EW.W_1[i][j] * EW.W_2[j][k]

        print(sequence)
        DET.tracking_sequence(sequence[0])
        # Reestart some variable for next iteration
        DET.list_results = []
        DET.list_YOLO_Camera_results = []
        # Object Detection Sequence
        # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
        object_detections = DET.final_results_clustering
        node_detection = dict_detections_nodes[id_seq]
        # Eval the weight
        EW.evaluate_location(3,
                             MAP.NODE_AN.LIST_NODES,
                             MAP.NODE_AN.LIST_NODES_EDGES,
                             node_detection,
                             object_detections,
                             sequence[1])

########################################################################################################################
########################################################################################################################
