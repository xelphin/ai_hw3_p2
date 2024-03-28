from ID3 import ID3
from utils import *
from DecisonTree import Leaf, Question, DecisionNode, class_counts


# File I created to test ID3

# TO RUN:
# python3 id3_test_main.py

def helper_print_counts_and_entropy(name, rows, labels, id3):
    parent_counts = class_counts(rows, labels)
    parent_node_entropy = id3.entropy(rows, labels)
    return f"{name} counts = {parent_counts} therefore, {name} entropy: {parent_node_entropy}"

def helper_print_tree(curr_node, str_path):
    if (isinstance(curr_node, Leaf)):
        print(f"{str_path}: is Leaf {isinstance(curr_node, Leaf)} of {curr_node.predictions}")
        return
    print(f"{str_path}: feature_{curr_node.question.column_idx} split {curr_node.question.value}")
    helper_print_tree(curr_node.true_branch, str_path + "->true")
    helper_print_tree(curr_node.false_branch, str_path + "->false")


def test_initiation(attributes_names):
    print("################################## test_initiation")
    id3 = ID3(attributes_names)

def test_entropy(attributes_names, x_train, y_train):
    print("################################## test_entropy")
    id3 = ID3(attributes_names)

    # We'll check the entropy of a node that has the first 10 people
    rows_arr = np.array(x_train[0:10]) # First 10 people data (excluding diagnosis)
    label_arr = np.array(y_train[0:10]) # First 10 people diagnosis data
    print(f"label_arr = {label_arr}")

    counts = class_counts(rows_arr, label_arr)
    print(f"counts = {counts}")

    # Check entropy calc
    entropy = id3.entropy(rows_arr, label_arr)
    print(f"Entropy = {entropy}")

def test_information_gain_aux(attributes_names, x_train, y_train, full_range, left_side, right_side):
    id3 = ID3(attributes_names)

    # We'll check the IG of splitting parent to two children: left and right
    full_rows_arr = np.array(x_train[full_range[0]:full_range[1]])
    full_label_arr = np.array(y_train[full_range[0]:full_range[1]])
    L_rows_arr = np.array(x_train[left_side[0]:left_side[1]])
    L_label_arr = np.array(y_train[left_side[0]:left_side[1]])
    R_rows_arr = np.array(x_train[right_side[0]:right_side[1]])
    R_label_arr = np.array(y_train[right_side[0]:right_side[1]])

    parent_node_entropy = id3.entropy(full_rows_arr, full_label_arr)

    print(helper_print_counts_and_entropy("Full",full_rows_arr, full_label_arr, id3))
    print(helper_print_counts_and_entropy("Left",L_rows_arr, L_label_arr, id3))
    print(helper_print_counts_and_entropy("Right",R_rows_arr, R_label_arr, id3))

    info_gain = id3.info_gain(L_rows_arr, L_label_arr, R_rows_arr, R_label_arr, parent_node_entropy)
    print(f"Info Gain = {info_gain}")

def test_information_gain_perfect_split(attributes_names, x_train, y_train):
    print("################################## test_information_gain_perfect_split")
    test_information_gain_aux(attributes_names, x_train, y_train, (0,6), (0,2), (3,6))

def test_information_gain_good_split(attributes_names, x_train,  y_train):
    print("################################## test_information_gain_good_split")
    test_information_gain_aux(attributes_names, x_train, y_train, (0,6), (0,3), (4,6))

def test_partition(attributes_names, x_train, y_train):
    print("################################## test_partition")
    id3 = ID3(attributes_names)

    # Lets take the first 12 people's smoothness mean
    smoothness_mean_col_index = 4
    rows_arr = np.array(x_train[0:12])
    label_arr = np.array(y_train[0:12])
    parent_entropy = id3.entropy(rows_arr, label_arr)

    # Lets make a question "smoothness_mean is >= 0.1"
    question = Question("smoothness_mean is >= ", smoothness_mean_col_index, 0.1)
    print(f"Question: {question}")

    # Lets see the partition it brings us
    gain, true_rows, true_labels, false_rows, false_labels = id3.partition(rows_arr, label_arr, question, parent_entropy)

    print(f"Gain: {gain}, amount true {true_rows.shape[0]}, amount false {false_rows.shape[0]}")
    print(f"True include {class_counts(true_rows, true_labels)}")
    print(f"False include {class_counts(false_rows, false_labels)}")
    

def test_find_best_split(attributes_names, x_train, y_train):
    print("################################## find_best_split")
    id3 = ID3(attributes_names)

    # Keep first 10 people and only first 3 features for simplicity in running this function
    rows = np.array(x_train[0:10, 0:3]) 
    labels = np.array(y_train[0:10])
    print(f"Testing on: \n{rows} \nLabels are: \n{labels}")
    best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = id3.find_best_split(rows, labels)

    print(f"Best gain {best_gain}")
    print(f"Best question {best_question}")
    print(f"True include {class_counts(best_true_rows, best_true_labels)}")
    print(f"False include {class_counts(best_false_rows, best_false_labels)}")

def test_build_tree(attributes_names, x_train, y_train):
    print("################################## test_build_tree")
    id3 = ID3(attributes_names)

    # Keep first 10 people and only first 3 features for simplicity in running this function
    rows = np.array(x_train[0:10, 0:3]) 
    labels = np.array(y_train[0:10])
    print(f"Testing on: \n{rows} \nLabels are: \n{labels}")

    root_node = id3.build_tree(rows, labels)
    helper_print_tree(root_node, "root")

'''
Node ['B' 'B' 'M' 'M' 'M' 'M' 'B' 'B' 'B' 'B']: feature_2 split 90.86 -> true ['M' 'M' 'M'] false ['B' 'B' 'M' 'B' 'B' 'B' 'B']
-LEAF M
-Node ['B' 'B' 'M' 'B' 'B' 'B' 'B']: feature_1 split 16.02 -> true ['B' 'B' 'B' 'B' 'B'] false ['B' 'M']
--LEAF B
--Node ['B' 'M']: feature_2 split 80.59 -> true ['M'] false ['B']
---LEAF M
---LEAF B
'''

def test_fit(attributes_names, x_train, y_train):
    print("################################## test_fit")
    print("[takes around 20sec...]")
    id3 = ID3(attributes_names)
    id3.fit(x_train, y_train)
    helper_print_tree(id3.tree_root, "root")


def test_predict_sample(attributes_names, x_train, y_train, x_test):
    print("################################## test_predict_sample")
    id3 = ID3(attributes_names)

    # Keep first 10 people and only first 3 features for simplicity in running this function
    rows = np.array(x_train[0:10, 0:3]) 
    labels = np.array(y_train[0:10])
    id3.fit(rows, labels)

    # Lets get the first person from the "test.csv"
    sample = np.array(x_test[0][0:3])
    print(f"Person data: {sample}")

    # Lets predict what their diagnosis is
    prediction = id3.predict_sample(sample)
    print(f"Prediction: {prediction}")

'''
Person data: [12.46 19.89 80.43]
At node where feature_2. Our value 80.43 >= 90.86 ? False
At node where feature_1. Our value 19.89 >= 16.02 ? True
At leaf, returning B
'''

def test_predict(attributes_names, x_train, y_train, x_test, y_test):
    print("################################## test_predict")
    id3 = ID3(attributes_names)

    # Keep first 10 people and only first 3 features for simplicity in testing
    rows = np.array(x_train[0:10, 0:3]) 
    labels = np.array(y_train[0:10])
    id3.fit(rows, labels)

    # Lets test out the first 20 people from "test.csv"
    sample_rows = np.array(x_test[0:20, 0:3]) 
    sample_predictions = id3.predict(sample_rows)
    sample_actual = y_test[0:20]

    # We got
    print(f"Sample predictions = {sample_predictions}")
    print(f"Sample actual      = {sample_actual}")
    print(f"Matches = {np.sum(sample_predictions == sample_actual)}")


if __name__ == '__main__':

    # Get data
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    target_attribute = 'diagnosis'
    (x_train, y_train, x_test, y_test) = get_dataset_split(train_dataset, test_dataset, target_attribute)

    # Tests
    print("-------------------------------------------")
    print("------------------ TESTS ------------------")
    print("-------------------------------------------")

    test_initiation(attributes_names)
    test_entropy(attributes_names, x_train, y_train)
    test_information_gain_perfect_split(attributes_names, x_train, y_train)
    test_information_gain_good_split(attributes_names, x_train, y_train)
    test_partition(attributes_names, x_train, y_train)
    test_find_best_split(attributes_names, x_train, y_train)
    test_build_tree(attributes_names, x_train, y_train)
    # test_fit(attributes_names, x_train, y_train)
    test_predict_sample(attributes_names, x_train, y_train, x_test)
    test_predict(attributes_names, x_train, y_train, x_test, y_test)
