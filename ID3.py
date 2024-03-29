import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        # Entropy over set E = H(E) = -sum(p(c_i)*log_2(p(c_i)))
        # p(c_i) of a node: (# objects of class i in node)/(# objects in node)
        amount_elems = labels.shape[0]
        for label in counts:
            p_label = counts[label]/amount_elems
            # print(f"p_{label} = {p_label}  , therefore add {-p_label*math.log2(p_label)}")
            impurity -= p_label*math.log2(p_label)
        
        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        size_of_both = left_labels.shape[0] + right_labels.shape[0]
        # Subtract left
        left_entropy = self.entropy(left, left_labels)
        left_size = left_labels.shape[0]
        left_value = (left_size/size_of_both)*left_entropy
        # Subtract right
        right_entropy = self.entropy(right, right_labels)
        right_size = right_labels.shape[0]
        right_value = (right_size/size_of_both)*right_entropy
        # Sum
        info_gain_value += current_uncertainty - left_value - right_value
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        true_rows = []
        true_labels = np.array([])
        false_rows = []
        false_labels = np.array([])
        total_people = rows.shape[0]
        for person_index in range(0, total_people):
            if question.match(rows[person_index]):
                true_rows += [rows[person_index]]
                true_labels = np.append(true_labels, labels[person_index])
                # print(f"True: Person{person_index} [{labels[person_index]}]: {rows[person_index]}")
            else:
                false_rows += [rows[person_index]]
                false_labels = np.append(false_labels, labels[person_index])
                # print(f"False: Person{person_index} [{labels[person_index]}]: {rows[person_index]}")

        true_rows = np.array(true_rows)
        false_rows = np.array(false_rows)
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels
    
    def helper_create_threshold_array(self, arr):
        thresholds = np.array([])

        for i in range(0, arr.shape[0]-1):
            threshold_val = (arr[i]+arr[i+1])/2
            thresholds = np.append(thresholds, threshold_val)

        return thresholds

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        
        if rows.shape[0] <= 1:
            # TODO (case only 1 or less people, how do you split?)
            return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

        # Iterate over features
        for feature_index in range(0, rows.shape[1]):
            # Get all people's value for that index to get thresholds
            peoples_values = np.sort(rows[:, feature_index])
            thresholds = self.helper_create_threshold_array(peoples_values)
            # print(f"feature_{feature_index} threshold: {thresholds}")
            # Iterate over thesholds
            for threshold in thresholds:
                # Create question
                question = Question(f"feature_{feature_index} >= ", feature_index, threshold)
                gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, question, current_uncertainty)
                if (gain >= best_gain):
                    best_gain = gain
                    best_question = question
                    best_true_rows = true_rows
                    best_true_labels = true_labels
                    best_false_rows = false_rows
                    best_false_labels = false_labels
                    # print(f"Found better with {question}")
                    # print(f"True: {class_counts(true_rows, true_labels)}\nFalse: {class_counts(false_rows, false_labels)}")
                    # print(f"Gain: {best_gain}")
        
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        
        # Recursion stop condition
        if len(np.unique(labels)) == 1:
            # Only one type of label for all people, so return a Leaf
            return Leaf(rows, labels) # labels shouldn't be empty
        
        # Min Pruning
        if rows.shape[0] <= self.min_for_pruning:
            return Leaf(rows, labels)
        
        # Find the best split
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)

        # Create children
        true_branch = self.build_tree(best_true_rows, best_true_labels)
        false_branch = self.build_tree(best_false_rows, best_false_labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        root = self.build_tree(x_train, y_train)
        self.tree_root = root
        # ========================



    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        # Case leaf
        if isinstance(node, Leaf):
            if len(node.predictions.keys()) == 0:
                return None
            max_key = max(node.predictions, key=node.predictions.get)
            # print(f"At leaf {node.predictions}, returning {max_key}")
            return max_key
        
        # Otherwise is DecisionNode
        feature_index = node.question.column_idx
        threshold_value = node.question.value
        our_value = row[feature_index]
        we_pass_question = our_value >= threshold_value
        # print(f"At node where feature_{feature_index}. Our value {our_value} >= {threshold_value} ? {we_pass_question}")
        if we_pass_question:
            return self.predict_sample(row, node.true_branch)
        return self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        amount_samples = rows.shape[0]
        y_pred = np.full(amount_samples, None)
        # print(f"We have {amount_samples} samples")
        for i in range(0, amount_samples):
            prediction = self.predict_sample(rows[i])
            y_pred[i] = prediction
        # ========================

        return y_pred
