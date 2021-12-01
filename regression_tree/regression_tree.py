import uuid
import graphviz

_INSERT, _DELETE = range(2)

class Leaf:
    
    def __init__(self, avarage_resdiual_image_vector):
        self.id = uuid.uuid4()
        self.avarage_residual_image_vector = avarage_resdiual_image_vector

    def get_node_description(self, detailed=False, root=False):
        return '[🍂 leaf, id: %s,\n -- avarage_residual_image_vector: %s]' % (self.id, self.avarage_residual_image_vector if detailed else self.avarage_residual_image_vector.shape)

    def get_dot_grahphviz_description(self, root=False):
        return '🍂 leaf, ⌀_residual_i_v: %s' % (str(self.avarage_residual_image_vector.shape))

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

    def get_node_description(self, detailed=False, root=False):
        return '[' + ('🌱 root' if root else '🔵 node') + ', id: %s, x1: %d, x2: %d, threshold: %d, left child id: %s, right child id: %s]' % (self.id, self.x1, self.x2, self.threshold, self.left_child_id, self.right_child_id)

    def get_dot_grahphviz_description(self, root=False):
        return '' + ('🌱 root' if root else '🔵 node') + ', x1: %d, x2: %d, threshold: %d' % (self.x1, self.x2, self.threshold)

class Regression_Tree:

    def __init__(self):
        self._nodes = []
        self._dot_graphviz = graphviz.Digraph('regression-tree', comment='A single regression tree')  

    def create_leaf(self, avarage_residual_image_vector, parent_id=None):
        leaf = Leaf(avarage_residual_image_vector)
        self._nodes.append(leaf)
        self.__update_childs(parent_id, leaf.id, _INSERT)
        self.update_dot_graphviz(leaf, parent_id)
        return leaf

    def create_node(self, x1, x2, threshold, parent_id=None):
        node = Node(x1, x2, threshold)
        self._nodes.append(node)
        self.__update_childs(parent_id, node.id, _INSERT)
        self.update_dot_graphviz(node, parent_id)
        return node

    def __update_childs(self, position, id, mode):
        if position is None:
            return
        else:
            if self[position].left_child_id == None:
                self[position].update_child(child_id=id, left=True, mode=mode)
            else:
                self[position].update_child(child_id=id, left=False, mode=mode)

    def get_tree_description(self, detailed=False):
        result = "<< 🌳 regression tree 🌳 >>\n\n"
        for index, node in enumerate(self._nodes):
            result += node.get_node_description(detailed, root=True if index == 0 else False) + "\n"
        return result + "\n<< 🌳 regression tree 🌳 >>\n"

    def _get_index(self, position):
        for index, node in enumerate(self._nodes):
            if node.id == position:
                break
        return index

    def __getitem__(self, key):
        return self._nodes[self._get_index(key)]

    def __setitem__(self, key, item):
        self._nodes[self._get_index(key)] = item

    def update_dot_graphviz(self, node, parent_id):
        self._dot_graphviz.node(str(node.id), node.get_dot_grahphviz_description(root=True if parent_id is None else False))
        if parent_id is not None:
            self._dot_graphviz.edge(str(parent_id), str(node.id))

    def get_dot_graphviz_source(self):
        return self._dot_graphviz.source