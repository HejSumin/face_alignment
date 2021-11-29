import numpy as np
import uuid
number_selected_features = 400

(_INSERT, _DELETE) = range(2)

class Node:

    def __init__(self, x1, x2, threshold):
        self.id = uuid.uuid4()
        self.x1 = x1
        self.x2 = x2
        self.threshold = threshold
        self.left_child_id = None
        self.right_child_id = None

    def update_child(self, child_id, left=True, mode=_INSERT):
        if mode is _INSERT:
            if left:
                self.left_child_id = child_id 
            else:
                self.right_child_id = child_id 
        elif mode is _DELETE:
            if left:
                self.left_child_id = None
            else:
                self.right_child_id = None

    def print_node(self):
        return '[id %s, x1 %d, x2 %d, threshold %d, left %s, right %s]' % (self.id, self.x1, self.x2, self.threshold, self.left_child_id, self.right_child_id)

class Tree:

    def __init__(self):
        self.nodes = []

    def create_node(self, x1, x2, threshold, parent_id=None):
        print("parent " + str(parent_id))
        node = Node(x1, x2, threshold)
        self.nodes.append(node)
        self.__update_childs(parent_id, node.id, _INSERT)
        return node

    def __update_childs(self, position, id, mode):
        if position is None:
            return
        else:
            if self[position].left_child_id == None:
                self[position].update_child(child_id=id, left=True, mode=mode)
            else:
                self[position].update_child(child_id=id, left=False, mode=mode)

    def print_tree(self):
        for node in self.nodes:
            print(node.print_node())

    def get_index(self, position):
        for index, node in enumerate(self.nodes):
            if node.id == position:
                break
        return index

    def __getitem__(self, key):
        return self.nodes[self.get_index(key)]

    def __setitem__(self, key, item):
        self.nodes[self.get_index(key)] = item


def generate_root(tree):
    random_x1_pixel_index_root = np.random.randint(0, number_selected_features)
    random_x2_pixel_index_root = np.random.randint(0, number_selected_features)
    while (random_x1_pixel_index_root == random_x2_pixel_index_root):
        random_x2_pixel_index_root = np.random.randint(0, number_selected_features)  

    random_threshold_root = np.random.randint(0, 255) # we take absolute value for node split pixel differnce

    root = tree.create_node(random_x1_pixel_index_root, random_x2_pixel_index_root, random_threshold_root)
    return root

def select_best_candidate_split_for_node(I, residual_image_vector, candidate_splits, Q): # I is [0, 255, 244, 123] grayscale values for 400 extracted pixels
    x1, x2, threshold = candidate_splits[0]
    Q_0_l = []
    Q_0_r = []
    for index in Q:
        if np.abs(I[index][x1] - I[index][x2]) > threshold:
            Q_0_l.append(index)
        else:
            Q_0_r.append(index)
    mu_0_l = 1 / len(Q_0_l) * np.sum(residual_image_vector[Q_0_l]) 
    mu_0_r = 1 / len(Q_0_r) * np.sum(residual_image_vector[Q_0_r])
    return

def generate_candidate_splits(number_candidate_splits=20):
    candidate_splits = []
    for i in range(0,number_candidate_splits-1): 
        random_x1_pixel_index = np.random.randint(0, number_selected_features)
        random_x2_pixel_index = np.random.randint(0, number_selected_features)
        while (random_x1_pixel_index == random_x2_pixel_index):
            random_x2_pixel_index = np.random.randint(0, number_selected_features)
        # generating 20 different x1, x2, threshold triplets and calcualting resiudual error here!

        random_threshold = np.random.randint(0, 255) # we take absolute value for node split pixel differnce
        candidate_splits.append(zip(random_x1_pixel_index, random_x2_pixel_index, random_threshold))
    return candidate_splits

def generate_complete_node(current_node_id, current_depth, max_depth):  
    if current_depth == max_depth:
        return
        
    random_x1_pixel_index_left_child = np.random.randint(0, number_selected_features)
    random_x2_pixel_index_left_child = np.random.randint(0, number_selected_features)
    while (random_x1_pixel_index_left_child == random_x2_pixel_index_left_child):
        random_x2_pixel_index_left_child = np.random.randint(0, number_selected_features)
    random_x1_pixel_index_right_child = np.random.randint(0, number_selected_features)
    random_x2_pixel_index_right_child = np.random.randint(0, number_selected_features)
    while (random_x1_pixel_index_right_child == random_x2_pixel_index_right_child):
        random_x2_pixel_index_right_child = np.random.randint(0, number_selected_features)

    random_threshold_left_child = np.random.randint(0, 255) # we take absolute value for node split pixel differnce
    random_threshold_right_child = np.random.randint(0, 255) # we take absolute value for node split pixel differnce

    left = tree.create_node(random_x1_pixel_index_left_child, random_x2_pixel_index_left_child, random_threshold_left_child, parent_id=current_node_id)  # node left, has parent current_node
    right = tree.create_node(random_x1_pixel_index_right_child, random_x2_pixel_index_right_child, random_threshold_right_child, parent_id=current_node_id)  # node right, has parent current_node

    return (generate_complete_node(left.id, current_depth+1, max_depth), generate_complete_node(right.id, current_depth+1, max_depth))

tree = Tree()
root = generate_root(tree)
generate_complete_node(current_node_id=root.id, current_depth=0, max_depth=2)

# print(len(tree.nodes))
# print(tree.print_tree())

index = [0,2]
array = np.array(["TRest", "tste", "no", "no", "no"])
print(array[index])