# from manual_tracking import ManualTrack
import copy
import shutil

from Tracking_Objects_v4 import Detection
from estimate_weights_v10_64Nodos import EstimationWeights
from map_annotation import Map

import numpy as np
import sys
import os


if (len(sys.argv)!=2):
    print("Arg error")
    print("Use python3 main_v2.py -eval")
    print("or")
    print("Use python3 main_v2.py -annotation")
    exit(0)

########################################################################################################################
# CAPTURA DE LOS ARGUMENTOS DE ENTRADA
########################################################################################################################
file_pruebas = open(os.path.join("Resources", "lista_pruebas.txt"))

# Check if flag annotation is active
if sys.argv[1] != "-annotation":
    lista_pruebas = []
    lista_nodos_pruebas = []
    for row in file_pruebas:

        # Split the data with white space
        aux_list = row.split("\n")[0].split(" ")

        for cont in range(0,len(aux_list)-1,4):
            # Reset the variables
            sequence_aux = []
            list_nodos_aux = []
            dict_aux = {}
            # Extract the nodes
            nodes_res_split = aux_list[cont].split("N")
            print("Nodo ini " + nodes_res_split[1])
            print("Nodo dest" + nodes_res_split[2])
            list_nodos_aux = [int(nodes_res_split[1]), int(nodes_res_split[2])]

            # Create the dictionary with data nodes class
            dict_aux = {
                'Node_init': aux_list[cont + 2],
                'Node_dest': aux_list[cont + 3],
                'dist': None
            }
        
            sequence_aux.append([aux_list[cont], float(aux_list[cont + 1]), dict_aux])
           
            lista_nodos_pruebas.append(list_nodos_aux)
            lista_pruebas.append(sequence_aux)
    # Eliminación del contenido de DiscretResults

    for files_dir in os.listdir(os.path.join("Results", "DiscretResults")):
        shutil.rmtree(os.path.join("Results", "DiscretResults", files_dir))

    # Eliminación del contenido de ResultsTracking
    for files_dir in os.listdir(os.path.join("Results", "ResultsTracking")):
        shutil.rmtree(os.path.join("Results", "ResultsTracking", files_dir))

    # Eliminación del contenido de la carpeta ResultsWeights
    for files_dir in os.listdir(os.path.join("Results", "ResultsWeights")):
        shutil.rmtree(os.path.join("Results", "ResultsWeights", files_dir))

    # Re-open the file pruebas
    file_pruebas = open(os.path.join("Resources", "lista_pruebas.txt"))
    # Creacion de las nuevas subcarpetas
    id_index = 0
    for __ in file_pruebas:
        file_dir = "Prueba_%d" % (id_index + 1)
        file_dir11 = "Prueba_%d" % (id_index + 1) +"/Tracking"

        # Creacion de las subcarpetas en DiscretResults
        os.mkdir(os.path.join("Results", "DiscretResults", file_dir))
        # Creacion de las subcarpetas en ResultsTracking
        os.mkdir(os.path.join("Results", "ResultsTracking", file_dir))
        os.mkdir(os.path.join("Results", "ResultsTracking", file_dir11))

        # Creacion de las subcarpetas en ResultsWeights
        os.mkdir(os.path.join("Results", "ResultsWeights", file_dir))
        # Incrementar el id_index
        id_index = id_index + 1



    """

    if sys.argv[1] != "-annotation":
        lista_pruebas = []
        lista_nodos_pruebas = []
        for row in file_pruebas:
            aux_list = row.split("\n")[0].split(" ")
            sequence_aux = []
            list_nodos_aux = []
            dict_aux = {}
            for ind_arr in range(0, len(aux_list), 4):
                print(ind_arr)
                print(row)
                # Lista de secuencia de nodos
                if ind_arr == 0:
                    list_nodos_aux.append(int(aux_list[ind_arr][5:7]))
                    list_nodos_aux.append(int(aux_list[ind_arr][8:]))
                else:
                    list_nodos_aux.append(int(aux_list[ind_arr][8:]))
                # Diccionario de la deteccion de nodos
                dict_aux = {
                    'Node_init': aux_list[ind_arr + 2],
                    'Node_dest': aux_list[ind_arr + 3],
                    'dist': None
                }
                sequence_aux.append([aux_list[ind_arr], float(aux_list[ind_arr + 1]), dict_aux])

            lista_nodos_pruebas.append(list_nodos_aux)
            lista_pruebas.append(sequence_aux)
    """
########################################################################################################################
# ELIMINACIÓN DE LOS FICHEROS DE RESULTADOS CREADOS PREVIOS A LA EJECUCIÓN DEL PROGRAMA
########################################################################################################################
elif (sys.argv[1] == "-annotation"):
    # Eliminación del contenido de DiscretResults

    for files_dir in os.listdir(os.path.join("Results", "DiscretResults")):
        shutil.rmtree(os.path.join("Results", "DiscretResults", files_dir))

    # Eliminación del contenido de ResultsTracking
    for files_dir in os.listdir(os.path.join("Results", "ResultsTracking")):
        shutil.rmtree(os.path.join("Results", "ResultsTracking", files_dir))

    # Eliminación del contenido de la carpeta ResultsWeights
    for files_dir in os.listdir(os.path.join("Results", "ResultsWeights")):
        shutil.rmtree(os.path.join("Results", "ResultsWeights", files_dir))

    # Re-open the file pruebas
    file_pruebas = open(os.path.join("Resources", "lista_pruebas.txt"))
    # Creacion de las nuevas subcarpetas
    id_index = 0
    for __ in file_pruebas:
        file_dir = "Prueba_%d" % (id_index + 1)
        file_dir11 = "Prueba_%d" % (id_index + 1) +"/Tracking"

        # Creacion de las subcarpetas en DiscretResults
        os.mkdir(os.path.join("Results", "DiscretResults", file_dir))
        # Creacion de las subcarpetas en ResultsTracking
        os.mkdir(os.path.join("Results", "ResultsTracking", file_dir))
        os.mkdir(os.path.join("Results", "ResultsTracking", file_dir11))

        # Creacion de las subcarpetas en ResultsWeights
        os.mkdir(os.path.join("Results", "ResultsWeights", file_dir))
        # Incrementar el id_index
        id_index = id_index + 1


def main():
    ####################################################################################################################
    # PARTE DEL TRACKING
    ####################################################################################################################
    # Object for detection (Tracking)
    DET = Detection("classes.names", "YOLO_v1")

   

    # Check if flag annotation is active
    if sys.argv[1] == "-annotation":
        print("MODE TRACKING ANNOTATION\n\n")

        id_sequence = 0
        file_pruebas = open(os.path.join("Resources", "lista_pruebas.txt"))
        P_edges_list = []

        for sequence in file_pruebas:
            sequence_split = sequence.split("\n")
            print(sequence_split[0])
            directory2save = "Prueba_%d" % (id_sequence + 1)
            distance_between_nodes = DET.tracking_sequence(sequence_split[0],
                                                           annotation=True,
                                                           path2save=directory2save)

            # Extract the nodes from sequence_split with a split
            nodes_split = sequence_split[0].split("N")

            # Save the data P_edges in a list
            P_edges_list.append([int(nodes_split[1]), int(nodes_split[2]), distance_between_nodes])

            # Reestart some variable for next iteration
            DET.list_results = []
            DET.list_YOLO_Camera_results = []

            id_sequence = id_sequence + 1

        # Create the txt P_edges.txt with the data distance from nodes
        np.savetxt("./Resources/Annotation_Tracking/P_edge.txt", np.array(P_edges_list), fmt="%d %d %.2f")
        """
        if sys.argv[1] == "-annotation":
            print("MODE TRACKING ANNOTATION\n\n")

            id_sequence = 0
            file_pruebas = open(os.path.join("Resources", "lista_pruebas.txt"))

            for sequence in file_pruebas:
                print(sequence)
                directory2save = "Prueba_%d" % (id_sequence + 1)
                distance_between_nodes = DET.tracking_sequence(sequence[0:10], annotation=True, path2save=directory2save)

                # Reestart some variable for next iteration
                DET.list_results = []
                DET.list_YOLO_Camera_results = []

                id_sequence = id_sequence + 1
        """
    elif (sys.argv[1] == "-eval"):
        print("MODE TRACKING AND EVAL POSE\n\n")

        ################################################################################################################
        # PARTE DE LA 'ANNOTATION'
        ################################################################################################################
        # Create the object from class Estimation
        EW = EstimationWeights("classes.names")
        # Create the annotations EDGES
        MAP = Map()

        ################################################################################################################
        # PARTE DE LA 'DETECTION'
        ################################################################################################################
        # Tracking and Evaluating for three sequences
        print("AAAA")
        print (lista_pruebas)
        for value_list, value_nodes_list, id_sequence in zip(lista_pruebas,
                                                             lista_nodos_pruebas,
                                                             range(len(lista_nodos_pruebas))):
            # Reinicio la variable list_data_representation de la clase EW
            EW.restart_list_data_representation()
            directory2save = "Prueba_%d" % (id_sequence + 1)
            for sequence, id_seq in zip(value_list, range(len(value_list))):
                if (id_seq+1) == 1:
                    print("ZZZZZZ")
                    print(sequence)
                    
                    distance_between_nodes = DET.tracking_sequence(sequence[0], path2save=directory2save,
                                                                   annotation=False)
                    sequence[2]['dist'] = distance_between_nodes

                    # Refresh the distance nodes with the results tracking
                    """first_node = int(sequence[0][5:7])
                    second_node = int(sequence[0][8:])

                    for value in MAP.NODE_AN.LIST_NODES_EDGES:
                        if value['Ref1'] == first_node and value['Ref2'] == second_node:
                            value['dist'] = distance_between_nodes
                            break
"""
                    # Reestart some variable for next iteration
                    DET.list_results = []
                    DET.list_YOLO_Camera_results = []
                    # Object Detection Sequence
                    # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
                    object_detections = DET.final_results_clustering
                    node_detection = sequence[2]
                    # Eval the weight
                    EW.evaluate_location(1,
                                         MAP.NODE_AN.LIST_NODES,
                                         MAP.NODE_AN.LIST_NODES_EDGES,
                                         node_detection,
                                         object_detections,
                                         sequence[1],
                                         path2save=directory2save,
                                         lista_nodos_prueba=value_nodes_list)

                    if id_sequence == len(lista_nodos_pruebas) - 1:
                        res_pond = round(EW.sumA - EW.sumB, 3)
                        msg2print = "\nPve = Sum(A) - Sum(B - 1)\n" + "Pve = " + str(res_pond)
                        EW.lista_pruebas_txt.write(msg2print)

                elif (id_seq+1) == 2:
                    print(sequence)
                    distance_between_nodes = DET.tracking_sequence(sequence[0], path2save=directory2save)
                    sequence[2]['dist'] = distance_between_nodes
                    # Reestart some variable for next iteration
                    DET.list_results = []
                    DET.list_YOLO_Camera_results = []
                    # Object Detection Sequence
                    # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
                    object_detections = DET.final_results_clustering
                    node_detection = sequence[2]
                    # Eval the weight
                    EW.evaluate_location(2,
                                         MAP.NODE_AN.LIST_NODES,
                                         MAP.NODE_AN.LIST_NODES_EDGES,
                                         node_detection,
                                         object_detections,
                                         sequence[1],
                                         path2save=directory2save,
                                         lista_nodos_prueba=value_nodes_list)

                    if id_sequence == len(lista_nodos_pruebas) - 1:
                        res_pond = round(EW.sumA - EW.sumB, 3)
                        msg2print = "\nPve = Sum(A) - Sum(B - 1)\n" + "Pve = " + str(res_pond)
                        EW.lista_pruebas_txt.write(msg2print)

                elif (id_seq + 1) == 3:
                    print(sequence)
                    distance_between_nodes = DET.tracking_sequence(sequence[0], path2save=directory2save)
                    sequence[2]['dist'] = distance_between_nodes
                    # Reestart some variable for next iteration
                    DET.list_results = []
                    DET.list_YOLO_Camera_results = []
                    # Object Detection Sequence
                    # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
                    object_detections = DET.final_results_clustering
                    node_detection = sequence[2]
                    # Eval the weight
                    EW.evaluate_location(3,
                                         MAP.NODE_AN.LIST_NODES,
                                         MAP.NODE_AN.LIST_NODES_EDGES,
                                         node_detection,
                                         object_detections,
                                         sequence[1],
                                         path2save=directory2save,
                                         lista_nodos_prueba=value_nodes_list)

                    if id_sequence == len(lista_nodos_pruebas) - 1:
                        res_pond = round(EW.sumA - EW.sumB, 3)
                        msg2print = "\nPve = Sum(A) - Sum(B - 1)\n" + "Pve = " + str(res_pond)
                        EW.lista_pruebas_txt.write(msg2print)

                if (id_seq + 1) > 3:
                    # In case to explore more than 3 edges
                    # Now the matrix_W1 will be the matrix_W2 from before iteration and the matrix W2 will be the
                    # matrix W3 from before iteration
                    EW.W_1 = copy.deepcopy(EW.W_2)
                    EW.W_2 = copy.deepcopy(EW.W_3)
                    # Now is calculated the new matrix W_12
                    # First restart the matrix W_12
                    EW.W_12 = np.zeros([64, 64, 64])

                    EW.W_12 = np.multiply(EW.W_1[:, :, np.newaxis], EW.W_2[np.newaxis, :])

                    print(sequence)
                    distance_between_nodes = DET.tracking_sequence(sequence[0], path2save=directory2save)
                    sequence[2]['dist'] = distance_between_nodes
                    # Reestart some variable for next iteration
                    DET.list_results = []
                    DET.list_YOLO_Camera_results = []
                    # Object Detection Sequence
                    # object_detections = MT.data_filter_objects(sequence[0]+"_Results.txt")
                    object_detections = DET.final_results_clustering
                    node_detection = sequence[2]
                    # Eval the weight
                    EW.evaluate_location(3,
                                         MAP.NODE_AN.LIST_NODES,
                                         MAP.NODE_AN.LIST_NODES_EDGES,
                                         node_detection,
                                         object_detections,
                                         sequence[1],
                                         path2save=directory2save,
                                         lista_nodos_prueba=value_nodes_list)

                    if id_sequence == len(lista_nodos_pruebas) - 1:
                        res_pond = round(EW.sumA - EW.sumB, 3)
                        msg2print = "\nPve = Sum(A) - Sum(B - 1)\n" + "Pve = " + str(res_pond)
                        EW.lista_pruebas_txt.write(msg2print)

        ################################################################################################################
        ################################################################################################################
    else:
        print("INCORRECT ARGUMENT\n")
        print("Use annotation=True for only create the map annotation\n")
        print("Use annotation=False if use the mode Tracking and Eval Pose")


if __name__ == "__main__":
    # Call the main function
    main()
