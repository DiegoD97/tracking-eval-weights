import os.path

from numpy import e
import numpy as np
import copy
from math import sqrt
from map_annotation import Map
import time
from statistics import median

MAP = Map()

class ResultsEvaluation(object):
    def __init__(self):
        # Attributes
        self.weights = []
        self.init_nodes = []
        self.dest_nodes = []
        self.possible_destinations = []
        self.Hist_Matrix_weights = []
        self.W_12 = []
        self.List_of_Matrix_Weights = []

    def empty_possible_destinations(self):
        if len(self.possible_destinations) == 0:
            return 1
        else:
            return 0

    def empty_historical_weights(self):
        if np.all(self.Hist_Matrix_weights == 0):
            return 1
        else:
            return 0


class EstimationWeights:
    ####################################################################################################################
    # BUILDER FOR CLASS
    ####################################################################################################################
    def __init__(self, file_ObjectClasses, topological_map=None):
        # Create the vars necessary for calculate the weights
        # LOAD THE OBJECT-CLASS LABELS OUR YOLO MODEL WAS TRAINED ON

        try:
            labelsPath = "Resources/" + file_ObjectClasses
            self.labels = open(labelsPath).read().strip().split("\n")
            self.n_classes = len(self.labels)  # Number of classes of objects
        except:
            print("File " + file_ObjectClasses + " is not found")
        # Constants gamma1 (objects) and gamma2 (nodes)
        self.gamma1 = 0.05
        self.gamma2 = 0.1
        self.gamma3 = 0.05

        # This dictionary is for save the data about the objects annotated that could be visualized with the
        # camera when the wheelchair navigate in the environment
        self.annotation_objects_vis = {
            'objects': {
                'window': {
                    'd_from_ref1': []
                },
                'door': {
                    'd_from_ref1': []
                },
                'elevator': {
                    'd_from_ref1': []
                },
                'fireext': {
                    'd_from_ref1': []
                },
                'plant': {
                    'd_from_ref1': []
                },
                'bench': {
                    'd_from_ref1': []
                },
                'firehose': {
                    'd_from_ref1': []
                },
                'lightbox': {
                    'd_from_ref1': []
                },
                'column': {
                    'd_from_ref1': []
                },
                'toilet': {
                    'd_from_ref1': []
                },
                'person': {
                    'd_from_ref1': []
                },
                'fallen': {
                    'd_from_ref1': []
                }
            }
        }

        # Create the Adjacency Matrix
        self.Matrix_Adyacencia = np.loadtxt('./Resources/EvalWeights/Matrix_Adyacencia_64.txt')
        # Create the Edges Matrix
        self.Matrix_Edges = np.loadtxt('./Resources/EvalWeights/Matrix_Edges.txt')
        shape_matrix = self.Matrix_Adyacencia.shape
        self.Matrix_Weights = np.zeros(shape_matrix)
        # Matrix Results for one, two and three edges
        self.W_1 = np.zeros([64, 64])
        self.W_2 = np.zeros([64, 64])
        self.W_3 = np.zeros([64, 64])
        self.W_12 = np.zeros([64, 64, 64])
        self.W_123 = np.zeros([64, 64, 64, 64])

        # Create the object for save the results from eval
        self.ResEval = ResultsEvaluation()

        # Variables de tiempo
        self.time_nodes_list = []
        self.time_objects_list = []
        self.time_Wi_list = []

        self.median_time_nodes = 0.0
        self.median_time_objects = 0.0
        self.median_time_matrix_Wi = 0.0

        # Variable for save the data to see from evaluation
        self.data_representation_Nodos = {
            'Contribucion nodos': None,
            'gamma Nodos': self.gamma1
        }
        self.data_representation_Objects = {
            'Clase objeto': None,
            'Contribucion clase objeto': None,
            'valor d': [],
            'valor dt': [],
            'gamma para d': self.gamma2,
            'gamma para dt': self.gamma3
        }
        self.data_representation_objects_edge = {
            'res objects': None,
            'NC': None,
            'peso': None
        }

        self.Matrix_RepResults = [[0] * 64 for z in range(64)]
        self.Matrix_ResNodes = [[0] * 64 for y in range(64)]

        self.list_data_representation = []
        # Umbral distance between the objects detected and annotated that will be used in function
        # for correspondences
        self.umbral = 1.5

        # Create a file for save the results from lista_pruebas.txt
        self.lista_pruebas_txt = open(os.path.join("Results", "results_lista_pruebas.txt"), "w")
        # Write the relevant data from class
        self.lista_pruebas_txt.write("gamma1 para nodos   %.3f\n" % self.gamma1)
        self.lista_pruebas_txt.write("gamma2 para d objetos  %.3f\n" % self.gamma2)
        self.lista_pruebas_txt.write("gamma3 para dt objetos   %.3f\n" % self.gamma3)
        self.lista_pruebas_txt.write("\n")

        labelsPathClust = "./Resources/umbral_clust.txt"
        for class_label, val_clust in zip(self.labels, open(labelsPathClust).read().strip().split("\n")):
            msg_aux = "u_clust  " + class_label + "  " + val_clust + "\n"
            self.lista_pruebas_txt.write(msg_aux)

        self.lista_pruebas_txt.write("\n")
        self.lista_pruebas_txt.write("Los valores que se indican a continuacion estan en el metodo weights_nodes\n")
        self.lista_pruebas_txt.write("N_init_an == N_init_det y N_dest_an == N_dest_det -> fj = 1\n")
        self.lista_pruebas_txt.write("N_init_an != N_init_det y N_dest_an == N_dest_det -> fj = 0.35\n")
        self.lista_pruebas_txt.write("N_init_an == N_init_det y N_dest_an != N_dest_det -> fj = 0.35\n")
        self.lista_pruebas_txt.write("N_init_an != N_init_det y N_dest_an != N_dest_det -> fj = 0.125\n\n")

        # Variable for evaluation
        self.A = 0.0
        self.B = 0.0
        self.sumA = 0.0
        self.sumB = 0.0

        # Load the annotated objects from topological map
        # self.annotated_objects = topological_map
    ####################################################################################################################
    # FUNCIONES PARA LIMPIAR LAS VARIABLES
    ####################################################################################################################
    def restart_data_representation_node(self):
        self.data_representation_Nodos = {
            'Contribucion nodos': None,
            'gamma Nodos': self.gamma1
        }

    def restart_data_representation_objects(self):
        self.data_representation_Objects = {
            'Clase objeto': None,
            'Contribucion clase objeto': None,
            'valor d': [],
            'valor dt': [],
            'gamma para d': self.gamma2,
            'gamma para dt': self.gamma3
        }

    def restart_data_representation_objects_edge(self):
        self.data_representation_objects_edge = {
            'res objects': None,
            'NC': None,
            'peso': None
        }

    def restart_list_data_representation(self):
        self.list_data_representation = []

    def copia_list_data_representation(self):
        return copy.deepcopy(self.list_data_representation)

    def copia_data_representation_node(self):
        return copy.deepcopy(self.data_representation_Nodos)

    def copia_data_representation_objects(self):
        return copy.deepcopy(self.data_representation_Objects)

    def copia_data_representation_objects_edge(self):
        return copy.deepcopy(self.data_representation_objects_edge)

    def copia_matriz_representacion_objetos(self):
        return copy.deepcopy(self.Matrix_RepResults)

    def copia_matriz_resultados_nodos(self):
        return copy.deepcopy(self.Matrix_ResNodes)

    def save_discrete_data_from_W(self):
        list_aux = [self.copia_matriz_resultados_nodos(), self.copia_matriz_representacion_objetos()]
        self.list_data_representation.append(list_aux)

    @staticmethod
    def represent_data_discreted(ruta_max_value, lista_data_representation, path2save, ruta_real=None):
        for index in range(len(lista_data_representation)):
            # Fichero para valor maximo de ruta
            filename_max = "./Results/DiscretResults/"+path2save+"/Nodes_%d_%d.txt" % (ruta_max_value[index]+1,
                                                                                       ruta_max_value[index+1]+1)
            filehandler_max = open(filename_max, 'wt')
            # Formato para los nodos
            data_weight_node = round(lista_data_representation[index][0][ruta_max_value[index]][ruta_max_value[index+1]]['Contribucion nodos'], 3)
            data_gamma_node = lista_data_representation[index][0][ruta_max_value[index]][ruta_max_value[index+1]]['gamma Nodos']
            format_node = "{0:<15}{1:<10}{2:<15}{3:<10}".format("Peso Nodos",
                                                                str(data_weight_node),
                                                                "gamma nodos",
                                                                str(data_gamma_node))
            filehandler_max.write("CONTRIBUCION NODOS\n")
            filehandler_max.write(format_node)

            filehandler_max.write("\n\nCONTRIBUCION OBJETOS\n")
            # Formato para los objetos
            for objClass in lista_data_representation[index][1][ruta_max_value[index]][ruta_max_value[index+1]]['res objects']:
                weight_obj_aux = str(round(objClass['Contribucion clase objeto'], 3))
                format_objects = "{0:<7}{1:<12}{2:<7}{3:<10}{4:<3}{5:<80}{6:<4}{7:<80}{8:<10}{9:<10}{10:<10}{11:<10}{12}".format(
                    "Clase:",
                    objClass['Clase objeto'],
                    "Peso:",
                    weight_obj_aux,
                    "d:",
                    str(objClass['valor d']),
                    "dt:",
                    str(objClass['valor dt']),
                    "gamma d:",
                    str(objClass['gamma para d']),
                    "gamma dt:",
                    str(objClass['gamma para dt']),
                    "\n")
                filehandler_max.write(format_objects)

            nc_aux = str(lista_data_representation[index][1][ruta_max_value[index]][ruta_max_value[index+1]]['NC'])
            peso_total_objs = round(lista_data_representation[index][1][ruta_max_value[index]][ruta_max_value[index+1]]['peso'], 3)
            peso_edge = round((data_weight_node + peso_total_objs) / 2, 3)
            format_other_param_objs = "{0:<10}{1:<10}{2}{3:<20}{4:<10}{5}{6:<10}{7}".format("NC:",
                                                                                            nc_aux,
                                                                                            "\n",
                                                                                            "Peso Total Objs:",
                                                                                            str(peso_total_objs),
                                                                                            "\n",
                                                                                            "Peso W1:",
                                                                                            str(peso_edge))
            filehandler_max.write(format_other_param_objs)
            filehandler_max.close()
            ##########################################################
            # Fichero para ruta real
            if ruta_real is not None:
                filename = "./Results/DiscretResults/"+path2save+"/Nodes_%d_%d.txt" % (ruta_real[index],
                                                                                       ruta_real[index + 1])
                filehandler = open(filename, 'wt')
                # Escritura para nodos
                data_node_weight = round(lista_data_representation[index][0][ruta_real[index]-1][ruta_real[index + 1]-1]['Contribucion nodos'], 3)
                data_node_gamma = lista_data_representation[index][0][ruta_real[index] - 1][ruta_real[index + 1] - 1]['gamma Nodos']
                format_node = "{0:<15}{1:<10}{2:<17}{3:<10}".format("Peso Nodos:",
                                                                    str(data_node_weight),
                                                                    "gamma nodos:",
                                                                    str(data_node_gamma))
                filehandler.write("CONTRIBUCION NODOS\n")
                filehandler.write(format_node)
                # Escritura para objetos
                filehandler.write("\n\nCONTRIBUCION OBJETOS\n")
                for objClass in lista_data_representation[index][1][ruta_real[index]-1][ruta_real[index + 1]-1]['res objects']:
                    weight_obj_aux = str(round(objClass['Contribucion clase objeto'], 3))
                    format_objects = "{0:<7}{1:<12}{2:<7}{3:<10}{4:<3}{5:<80}{6:<4}{7:<80}{8:<10}{9:<10}{10:<10}{11:<10}{12}".format("Clase:",
                                                                                                                                   objClass['Clase objeto'],
                                                                                                                                   "Peso:",
                                                                                                                                   weight_obj_aux,
                                                                                                                                   "d:",
                                                                                                                                   str(objClass['valor d']),
                                                                                                                                   "dt:",
                                                                                                                                   str(objClass['valor dt']),
                                                                                                                                   "gamma d:",
                                                                                                                                   str(objClass['gamma para d']),
                                                                                                                                   "gamma dt:",
                                                                                                                                   str(objClass['gamma para dt']),
                                                                                                                                   "\n")

                    filehandler.write(format_objects)

                nc_aux = str(lista_data_representation[index][1][ruta_real[index]-1][ruta_real[index + 1]-1]['NC'])
                peso_total_objs_r = round(lista_data_representation[index][1][ruta_real[index]-1][ruta_real[index + 1]-1]['peso'], 3)
                peso_edge_r = round((data_node_weight + peso_total_objs_r) / 2, 3)
                format_other_param_objs = "{0:<10}{1:<10}{2}{3:<20}{4:<10}{5}{6:<10}{7}".format("NC:",
                                                                                            nc_aux,
                                                                                            "\n",
                                                                                            "Peso Total Objs:",
                                                                                            str(peso_total_objs_r),
                                                                                            "\n",
                                                                                            "Peso W1:",
                                                                                            str(peso_edge_r))
                filehandler.write(format_other_param_objs)
                filehandler.close()

        lista_data_representation.pop(0)

    ####################################################################################################################
    # FUNCTION FOR REPRESENT THE DATA FROM MATRIX W_1, W_2, W_3, W_12, W_123
    ####################################################################################################################
    @staticmethod
    def represent_data_matrix(file_name, matrix2rep, min_statistic, path2save):
        file_W_txt = open('./Results/ResultsWeights/'+path2save+'/'+file_name+".txt", 'w')
        for row in range(matrix2rep.shape[0]):
            val_max = np.max(matrix2rep[row])
            if val_max > min_statistic:
                position = np.where(matrix2rep[row] == val_max)
                if len(matrix2rep.shape) == 2:
                    result = file_name + "(%d, %d)" % (row + 1, position[0] + 1) + " = " + str(round(val_max, 3)) + "\n"
                elif len(matrix2rep.shape) == 3:
                    result = "W_12" + "(%d, %d, %d)" % (row + 1, position[0][0] + 1, position[1][0] + 1) + " = " + str(
                        round(val_max, 3)) + "\n"
                elif len(matrix2rep.shape) == 4:
                    result = ("W_123" + "(%d, %d, %d, %d)" % (row + 1,
                                                              position[0][0] + 1,
                                                              position[1][0] + 1,
                                                              position[2][0] + 1) + " = " + str(
                        round(val_max, 3)) + "\n")
                # Write in the file
                file_W_txt.write(result)
            else:
                pass

    ####################################################################################################################
    # FUNCTION FOR EVALUATE THE WEIGHT (CORRESPONDENCE BETWEEN DETECTION AND ANNOTATION) FROM OBJECTS DETECTED
    ####################################################################################################################

    def weights_objects(self, edge_annotation, edge_detection, direction, dist_nodes_edge):
        # FIRST: in local var save the objects that the platform see. Depending on your movement
        # could not detect all the objects from annotation, so it is necessary filter the annotation
        # objects for the possible detection.
        try:
            # Restart la variable de data_representation_objects_edge
            self.restart_data_representation_objects_edge()

            self.annotation_objects_vis = edge_annotation
            # Local var like security copy
            copy_annotation_objects_vis = copy.deepcopy(self.annotation_objects_vis)
            copy_edge_detection = copy.deepcopy(edge_detection)
            # Loop for calculate the weight of objects
            # Init the result product
            result_product = 1
            res_objects_for_edge = []
            NC = 0
            for classObject_an in copy_annotation_objects_vis['objects']:
                for classObject_det in copy_edge_detection:
                    # Case to detect the same object that in annotation
                    if classObject_det == classObject_an:
                        # Check the lengths of list objects detected and annotated
                        if (len(copy_annotation_objects_vis['objects'][classObject_an]['d_from_ref1']) <=
                                len(copy_edge_detection[classObject_det]['d_from_ref1'])):
                            num_iter = len(copy_annotation_objects_vis['objects'][classObject_an]['d_from_ref1'])
                        else:
                            num_iter = len(copy_edge_detection[classObject_det]['d_from_ref1'])

                        result_product_partial = 1.0

                        for __ in range(num_iter):
                            # Study the correlation object
                            P1, P2, d, dt = self.object_correlation(copy_annotation_objects_vis['objects'][classObject_an],
                                                                    copy_edge_detection[classObject_det],
                                                                    direction,
                                                                    dist_nodes_edge)
                            # Comprobate if the euclidean distance from d and dt is over an umbral distance
                            # if sqrt(d ** 2 + dt ** 2) > self.umbral:
                            #    pass
                            # else:
                            # Guardar las distancias de d y dt en el data representation
                            self.data_representation_Objects['valor d'].append(round(d, 3))
                            self.data_representation_Objects['valor dt'].append(round(dt, 3))
                            # Eval the weight using the distance from node reference and transversal distance
                            result_product_partial = (result_product_partial *
                                                      (e ** (-self.gamma1 * d)) * (e ** (-self.gamma3 * dt)))

                            result_product = result_product * (e ** (-self.gamma1 * d)) * (e ** (-self.gamma3 * dt))
                            # result_product = result_product * (e ** (-self.gamma1 * d))
                            # Delete the value studied for next iteration
                            copy_annotation_objects_vis['objects'][classObject_an]['d_from_ref1'].pop(P2)
                            copy_annotation_objects_vis['objects'][classObject_an]['d_trans'].pop(P2)
                            # Delete the value studied for next iteration
                            copy_edge_detection[classObject_det]['d_from_ref1'].pop(P1)
                            copy_edge_detection[classObject_det]['d_trans'].pop(P1)

                        # Sacar resultados
                        if num_iter != 0:
                            self.data_representation_Objects['Clase objeto'] = classObject_det
                            self.data_representation_Objects['Contribucion clase objeto'] = result_product_partial

                            res_objects_for_edge.append(self.copia_data_representation_objects())
                        self.restart_data_representation_objects()

            # Finally, the objects of non-correspondence in the annotation and in the detection are counted.
            # NC for annotation
            for Object_an in copy_annotation_objects_vis['objects']:
                if len(copy_annotation_objects_vis['objects'][Object_an]['d_from_ref1']) != 0:
                    NC += len(copy_annotation_objects_vis['objects'][Object_an]['d_from_ref1'])
            # NC for detection
            for Object_det in copy_edge_detection:
                if len(copy_edge_detection[Object_det]['d_from_ref1']) != 0:
                    NC += len(copy_edge_detection[Object_det]['d_from_ref1'])

            # Guardar los objetos no relacionados
            self.data_representation_objects_edge['NC'] = NC

            # Calculate the result of products
            result_product = result_product * (0.8 ** NC)

            # Guardar el peso final y los parciales
            self.data_representation_objects_edge['peso'] = result_product
            self.data_representation_objects_edge['res objects'] = res_objects_for_edge

            # Free the data from self.annotation_objects_vis for next iteration
            for ObjectVis in self.annotation_objects_vis['objects']:
                while (len(self.annotation_objects_vis['objects'][ObjectVis]['d_from_ref1'])) != 0:
                    self.annotation_objects_vis['objects'][ObjectVis]['d_from_ref1'].pop(-1)
                    self.annotation_objects_vis['objects'][ObjectVis]['d_trans'].pop(-1)

        except TypeError:
            result_product = 0

        return result_product, self.data_representation_objects_edge

    ####################################################################################################################
    # FUNCTION FOR EVALUATE THE WEIGHT FROM NODES DETECTED
    ####################################################################################################################

    def weights_nodes(self, eval_node_init, eval_node_dest, node_annotation_init, node_annotation_dest, edge_annotated,
                      node_detect):
        try:
            # Check if the node detected and annotated at the init are the same
            if node_detect['Node_init'] == node_annotation_init['class']:
                # Now check if the node destiny are the same
                if node_detect['Node_dest'] == node_annotation_dest['class']:
                    fj = 1
                else:
                    fj = 0.35
            # In otherwise, where the init node not correspondence with the detection
            else:
                # In case that the final node correspondence between the detection and annotation
                if node_detect['Node_dest'] == node_annotation_dest['class']:
                    fj = 0.35
                else:
                    fj = 0.125
            # Determine the distance between nodes
            if ((edge_annotated['Ref1'] == eval_node_init and edge_annotated['Ref2'] == eval_node_dest) or
                (edge_annotated['Ref1'] == eval_node_dest and edge_annotated['Ref2'] == eval_node_init)):

                # Distance between nodes
                d_annotation = edge_annotated['dist']
                # Select the direction of movement
                if edge_annotated['Ref1'] == eval_node_init:
                    direction_is = 1
                else:
                    direction_is = -1
            else:
                # Init the var d_annotation for having the maximum error
                d_annotation = 0
                direction_is = None

            distance = abs(node_detect['dist'] - d_annotation)

            result_product = fj * (e ** (-self.gamma2 * distance))

        except TypeError:
            result_product = 0
            direction_is = None

        return result_product, direction_is

    ####################################################################################################################
    # FUNCTION FOR EVALUATE THE TOTAL WEIGHT CONSIDERING THE OBJECTS_WEIGHT AND NODES_WEIGHT
    ####################################################################################################################

    def evaluate_weight(self, Nodes_Annotated, Edges_Annotated, Nodes_Detect, Objects_Detect, NumIter, Ang_g):
        # Restart the matrix weights
        self.Matrix_Weights = np.zeros((64, 64))
        self.Matrix_RepResults = [[0] * 64 for z in range(64)]
        self.Matrix_ResNodes = [[0] * 64 for y in range(64)]
        # Loop
        for i in Nodes_Annotated:
            for j in Nodes_Annotated:
                if ((i != j) and (self.Matrix_Adyacencia[i - 1][j - 1] == 1) and
                   self.compatibility_of_turns(Ang_g, i, j, NumIter)):

                    edge = int(self.Matrix_Edges[i - 1][j - 1]) - 1
                    # OJO: TIEMPO 3
                    t3 = time.time()
                    weight_nodes, direction_sel = self.weights_nodes(i, j,
                                                                     Nodes_Annotated[i],
                                                                     Nodes_Annotated[j],
                                                                     Edges_Annotated[edge],
                                                                     Nodes_Detect)

                    self.data_representation_Nodos['Contribucion nodos'] = weight_nodes
                    # OJO: TIEMPO 4
                    t4 = time.time()
                    self.time_nodes_list.append(t4 - t3)
                    weight_objects, data_results_objs = self.weights_objects(Edges_Annotated[edge], Objects_Detect,
                                                                             direction_sel, Nodes_Detect['dist'])

                    # OJO: TIEMPO 5
                    t5 = time.time()
                    self.time_objects_list.append(t5 - t4)

                    total_weight = 0.5 * (weight_nodes + weight_objects)
                    self.Matrix_Weights[i - 1][j - 1] = total_weight
                    self.Matrix_RepResults[i - 1][j - 1] = data_results_objs
                    self.Matrix_ResNodes[i - 1][j - 1] = self.copia_data_representation_node()

        return self.Matrix_Weights

    ####################################################################################################################
    # FUNCTION FOR EVAL THE LOCATION USING THE INFORMATION FROM 'n'-EDGES
    ####################################################################################################################

    def evaluate_location(self, Num_Edges_Eval, Nodes_Annotated, Edges_Annotated, Nodes_Detect, Objects_Detect, Ang_g,
                          path2save, lista_nodos_prueba):
        # Create a deepcopy for variable Edges_Annotated
        Copy_Edges_Annotated = copy.deepcopy(Edges_Annotated)
        # FIRST: Evaluate the weight from each Edge for actual detection
        # OJO: TIEMPO 1
        t1 = time.time()
        W = self.evaluate_weight(Nodes_Annotated,
                                 Copy_Edges_Annotated,
                                 Nodes_Detect,
                                 Objects_Detect,
                                 Num_Edges_Eval,
                                 Ang_g)

        # HAGO UNA COPIA DE LAS MATRICES RESULTADOS Y LAS ALMACENO EN UNA LISTA
        self.save_discrete_data_from_W()
        # OJO: TIEMPO 2
        t2 = time.time()
        self.time_Wi_list.append(t2 - t1)

        # SECOND: comprobate the idEdge
        if Num_Edges_Eval == 1:
            self.W_1 = copy.deepcopy(W)
            # Final Result
            # Most relevant values for W_1
            print("\n\n\nMAX VALUES FOR MATRIX W_1")
            caminos_probables = self.max_values_from_matrix(3, self.W_1)
            self.represent_data_discreted(caminos_probables[0], self.copia_list_data_representation(), path2save)
            t1_t2 = t2 - t1
            print("\nTiempo de ejecución de la matriz W1: %f" % t1_t2)
            # Save data matrix
            self.represent_data_matrix("W_1", self.W_1, 0.4, path2save)
            # Comprobar si la secuencia de lista pruebas tiene solo un edge a evaluar (2 nodos)
            if len(lista_nodos_prueba) == 2:
                if lista_nodos_prueba == [y+1 for y in caminos_probables[0]]:
                    W_max = self.W_1[caminos_probables[0][0],
                                     caminos_probables[0][1]]
                    W_max_2 = self.W_1[caminos_probables[1][0],
                                       caminos_probables[1][1]]

                    Value_Ponderado = round(W_max / W_max_2, 3)
                    self.sumA += Value_Ponderado
                    self.resultados_de_la_lista_de_pruebas(caminos_probables[0], " C ", Value_Ponderado)
                else:
                    W_max = self.W_1[caminos_probables[0][0],
                                     caminos_probables[0][1]]
                    W_max_2 = self.W_1[lista_nodos_prueba[0]-1,
                                       lista_nodos_prueba[1]-1]

                    Value_Ponderado = round(W_max / W_max_2, 3)
                    self.sumB += (Value_Ponderado - 1)
                    self.resultados_de_la_lista_de_pruebas(caminos_probables[0], " I ", Value_Ponderado,
                                                           lista_nodos_prueba)

        elif Num_Edges_Eval == 2:
            # In this case is necessary evaluate the combinations for two Edges
            for i in range(self.W_1.shape[0]):
                for j in range(self.W_1.shape[1]):
                    for k in range(self.W_1.shape[1]):
                        self.W_12[i][j][k] = self.W_1[i][j] * W[j][k]

            # OJO TIEMPO 6: Tiempo de ejecucion para la matriz W12
            t6 = time.time()
            # Save the max values sequences for matrix W12
            self.represent_data_matrix("W_12", self.W_12, 0.1, path2save)

            # Most relevant values for W_12
            print("\n\n\nMAX VALUES FOR MATRIX W_12")
            caminos_probables = self.max_values_from_matrix(3, self.W_12)
            self.represent_data_discreted(caminos_probables[0], self.copia_list_data_representation(), path2save)
            print("\n\nTIEMPOS DE EJECUCIÓN DEL PROGRAMA")
            t6_t2 = t6 - t2
            print("\nTiempo de ejecución de la matriz W12: %f" % t6_t2)
            # Save the matrix for second Edge
            self.W_2 = copy.deepcopy(W)
            # Save data matrix
            self.represent_data_matrix("W_2", self.W_2, 0.4, path2save)
            # Comprobar si la secuencia de lista pruebas tiene dos edge a evaluar (3 nodos)
            if len(lista_nodos_prueba) == 3:
                if lista_nodos_prueba == [y+1 for y in caminos_probables[0]]:
                    W_max = self.W_12[caminos_probables[0][0],
                                      caminos_probables[0][1],
                                      caminos_probables[0][2]]
                    W_max_2 = self.W_12[caminos_probables[1][0],
                                        caminos_probables[1][1],
                                        caminos_probables[1][2]]

                    Value_Ponderado = round(W_max / W_max_2, 3)
                    self.sumA += Value_Ponderado
                    self.resultados_de_la_lista_de_pruebas(caminos_probables[0], " C ", Value_Ponderado)
                else:
                    W_max = self.W_12[caminos_probables[0][0],
                                      caminos_probables[0][1],
                                      caminos_probables[0][2]]
                    W_max_2 = self.W_12[lista_nodos_prueba[0]-1,
                                        lista_nodos_prueba[1]-1,
                                        lista_nodos_prueba[2]-1]

                    Value_Ponderado = round(W_max / W_max_2, 3)
                    self.sumB += (Value_Ponderado - 1)
                    self.resultados_de_la_lista_de_pruebas(caminos_probables[0], " I ", Value_Ponderado,
                                                           lista_nodos_prueba)

        elif Num_Edges_Eval == 3:
            for n in range(self.W_12.shape[0]):
                for m in range(self.W_12.shape[1]):
                    for o in range(self.W_12.shape[2]):
                        self.W_123[n][m][o][:] = self.W_12[n][m][o] * W[o][:]

            # OJO: TIEMPO 7
            t7 = time.time()
            t7_t6 = t7 - t2
            # Save the max values sequences for matrix W123
            self.represent_data_matrix("W_123", self.W_123, 0.1, path2save)
            # OJO: TIEMPO 8
            t8 = time.time()
            t8_t7 = t8 - t7

            # Save the matrix for third Edge
            self.W_3 = copy.deepcopy(W)
            self.represent_data_matrix("W_3", self.W_3, 0.4, path2save)

            # Most relevant values for W_123
            print("\n\n\nMAX VALUES FOR MATRIX W_123")
            caminos_probables = self.max_values_from_matrix(3, self.W_123)
            ############################################################################################################
            # Valores discretizados a imprimir
            """
            camino_real = []
            for w in range(4):
                camino_real.append(int(input("Nodo %d: " % (w+1))))

            position = np.where(self.W_123 == np.max(self.W_123))
            camino_max_value = [position[0][0], position[1][0], position[2][0], position[3][0]]
            """
            self.represent_data_discreted(caminos_probables[0], self.copia_list_data_representation(), path2save)
            # Finalmente despues de evaluar tres edges consecutivos se reinicia la variable list_data_representation
            self.restart_list_data_representation()
            # Comprobar si la secuencia de lista pruebas tiene tres edge a evaluar (4 nodos)
            if len(lista_nodos_prueba) == 4:
                if lista_nodos_prueba == [y+1 for y in caminos_probables[0]]:
                    W_max = self.W_123[caminos_probables[0][0],
                                       caminos_probables[0][1],
                                       caminos_probables[0][2],
                                       caminos_probables[0][3]]
                    W_max_2 = self.W_123[caminos_probables[1][0],
                                         caminos_probables[1][1],
                                         caminos_probables[1][2],
                                         caminos_probables[1][3]]

                    Value_Ponderado = round(W_max / W_max_2, 3)
                    self.sumA += Value_Ponderado
                    self.resultados_de_la_lista_de_pruebas(caminos_probables[0], " C ", Value_Ponderado)
                else:
                    W_max = self.W_123[caminos_probables[0][0],
                                       caminos_probables[0][1],
                                       caminos_probables[0][2],
                                       caminos_probables[0][3]]
                    W_max_2 = self.W_123[lista_nodos_prueba[0]-1,
                                         lista_nodos_prueba[1]-1,
                                         lista_nodos_prueba[2]-1,
                                         lista_nodos_prueba[3]-1]

                    Value_Ponderado = round(W_max / W_max_2, 3)
                    self.sumB += (Value_Ponderado - 1)
                    self.resultados_de_la_lista_de_pruebas(caminos_probables[0], " I ", Value_Ponderado,
                                                           lista_nodos_prueba)
            ############################################################################################################
            print("\n\nTIEMPOS DE EJECUCIÓN DEL PROGRAMA")
            print("\nTiempo de ejecución de la matriz W123: %f" % t7_t6)
            print("\nTiempo de busqueda de maximos en W123: %f" % t8_t7)

        # Determinacion del tiempo medio de ejecucion de nodos y objetos
        print("\n\nTiempo de ejecución de los nodos: %f" % median(self.time_nodes_list))
        print("\nTiempo de ejecución de los objetos: %f" % median(self.time_objects_list))

    ####################################################################################################################
    # FUNCTION FOR DETERMINE THE CORRELATION BETWEEN OBJECTS
    ####################################################################################################################
    @staticmethod
    def object_correlation(annotated_object, detected_object, direction, distance):
        # Local variables
        # The vars P1 and P2 contain the index from objects of same class with the minimum distance
        P1 = []
        P2 = []
        # Initialize the var error distance
        err_d_euclidean = 1000
        err_d_result = 1000
        err_t_result = 1000

        # Loop for travel the dictionary from detection
        for index_obj_det in range(len(detected_object['d_from_ref1'])):
            # Loop for travel the dictionary from annotation
            for index_obj_an in range(len(annotated_object['d_from_ref1'])):
                # Observate the direction of platform
                if direction == 1:  # The platform goes from NodeRef1 to NodeRef2
                    dist_annotation = annotated_object['d_from_ref1'][index_obj_an]
                    dist_trans_annotation = annotated_object['d_trans'][index_obj_an]
                elif direction == -1:
                    dist_annotation = (distance - annotated_object['d_from_ref1'][index_obj_an])
                    dist_trans_annotation = annotated_object['d_trans'][index_obj_an]
                else:
                    break

                # Calculate the error distance between the distance from annotation and detection
                err_d = abs(detected_object['d_from_ref1'][index_obj_det] - dist_annotation)
                # Calculate the error transversal distance
                err_t = abs(detected_object['d_trans'][index_obj_det] - dist_trans_annotation)
                # Calculate the error distance euclidean
                err_euclidean = sqrt(err_d ** 2 + err_t ** 2)

                # Case the minimum error for frontal distance
                if err_euclidean < err_d_euclidean:
                    # Euclidean distance error
                    err_d_euclidean = err_euclidean
                    # Results for return
                    err_d_result = err_d
                    err_t_result = err_t
                    # Var for detection
                    P1 = index_obj_det
                    # Var for annotation
                    P2 = index_obj_an

        return P1, P2, err_d_result, err_t_result

    ########################################################################
    # VERIFICATION IN THE COMPATIBILITY OF TURNS
    ########################################################################
    @staticmethod
    def compatibility_of_turns(last_turn, node1, node2, numIter):
        """
        This function comprobate the compatibility of turns
        :param last_turn: lecture of turn angle from odometry
        :param node1: the node to study
        :param node2: the destiny node
        :param numIter: number of iteration
        :return: value boolean '0': not exist compatibility '1': exist compatibility
        """
        # First Edge
        if numIter == 1:
            # Exist compatibility with all of Edges
            return 1
        # Comprobate the turns
        else:
            # Node destiny
            index_dest = np.where(np.array(MAP.NODE_AN.LIST_NODES[node1]['Node']) == node2)
            index_dest = int(index_dest[-1])
            ang_dest = MAP.NODE_AN.LIST_NODES[node1]['Ang'][index_dest]
            ####################################
            # Initial Node when node1 is class T
            ####################################
            if MAP.NODE_AN.LIST_NODES[node1]['class'] == 'T':
                for node in MAP.NODE_AN.LIST_NODES[node1]['Node']:
                    if node == node2:
                        pass
                    else:
                        index_origin = np.where(np.array(MAP.NODE_AN.LIST_NODES[node1]['Node']) == node)
                        index_origin = int(index_origin[-1])
                        ang_origin = MAP.NODE_AN.LIST_NODES[node1]['Ang'][index_origin]
                        # Result of annotation
                        ang_annotated = ang_dest - ang_origin
                        # Test if the angle annotated is the -270 degree, in this case the angle annotated is 90 degree
                        if ang_annotated < 0 and (270 - abs(ang_annotated)) < 10.0:
                            ang_annotated = 90.0
                        # Test if the result of angle annotated is +-180 degree and the increment
                        # of angular turned is 0 degree
                        if last_turn < 10.0 and (180.0 - abs(ang_annotated)) < 10.0:
                            ang_annotated = 0.0
                        # Test if the result of angle annotated is 0 degree and the increment of
                        # angular turned is +-180 degree
                        elif abs(last_turn) > 170.0 and ang_annotated < 10.0:
                            last_turn = 0.0
                        # Test if result of angle annotated is +-180 degree and the increment of
                        # angular turned is +-180 degree
                        elif abs(last_turn) > 170.0 and (180.0 - abs(ang_annotated)) < 10.0:
                            last_turn = 0.0
                            ang_annotated = 0.0
                        # Comprobate the compatibility turning
                        if abs(last_turn - ang_annotated) < 30.0:
                            return 1
                # If the loop is traveled and is not return the compatibility, it does not exist compatibility
                return 0
            ####################################
            # Initial Node when node1 is class E
            ####################################
            elif MAP.NODE_AN.LIST_NODES[node1]['class'] == 'E':
                # In this case the node origin and the node to study (node1) are the same
                ang_annotated = ang_dest
                # Test if the angle annotated is 0 degree and the increment of angular turned is +-180 degree
                if abs(last_turn) > 170.0 and ang_annotated < 10.0:
                    last_turn = 0.0
                # Finally, comprobate the compatibility turning
                if abs(last_turn - ang_annotated) < 30.0:
                    return 1
                else:
                    return 0

    ###################################################
    # FUNCTION FOR DETERMINE THE MAX VALUES FROM MATRIX
    ###################################################
    @staticmethod
    def max_values_from_matrix(num_values, matrix_data):
        # Create a deepcopy
        matrix_aux = copy.deepcopy(matrix_data)
        camino_mas_probable = []

        for n in range(num_values):
            res = np.where(matrix_aux == np.max(matrix_aux))
            pos_id_max = []
            for p in res:
                pos_id_max.append(p[0])

            camino_mas_probable.append(pos_id_max)

            sec = ""
            for q in range(len(pos_id_max)):
                if q != (len(pos_id_max) - 1):
                    sec += "%d => " % (pos_id_max[q] + 1)
                else:
                    sec += "%d " % (pos_id_max[q] + 1)

            print("\n\nEl recorrido más probable es:")
            print(sec)
            print("Con un peso de W = %.4f" % np.max(matrix_aux))
            pos_id_max = tuple(pos_id_max)
            matrix_aux.itemset(pos_id_max, 0.0)

        return camino_mas_probable

    ##################################################################
    # FUNCTION FOR CREATE THE .txt WITH RESULTS OF LIST TEST SEQUENCES
    ##################################################################
    def resultados_de_la_lista_de_pruebas(self, way_most_probable, msg_st, value_pond, camino_real=None):
        msg = str(value_pond) + msg_st

        if camino_real is not None:
            sec = ""
            for ind in range(len(way_most_probable)):
                if ind != (len(way_most_probable) - 1):
                    sec += "%d => " % (way_most_probable[ind] + 1)
                else:
                    sec += "%d" % (way_most_probable[ind] + 1)

            sec_real = ""
            for ind in range(len(camino_real)):
                if ind != (len(camino_real) - 1):
                    sec_real += "%d => " % (camino_real[ind])
                else:
                    sec_real += "%d" % (camino_real[ind])

            msg = msg + sec + " (Camino Real: " + sec_real + ")\n"

        else:
            sec = ""
            for ind in range(len(way_most_probable)):
                if ind != (len(way_most_probable) - 1):
                    sec += "%d => " % (way_most_probable[ind] + 1)
                else:
                    sec += "%d\n" % (way_most_probable[ind] + 1)

            msg = msg + sec

        self.lista_pruebas_txt.write(msg)

########################################################################################################################
# END CLASS
########################################################################################################################
