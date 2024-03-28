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

    # Keep first 10 people and only first 3 features for simplicity in testing
    rows = np.array(x_train[0:10, 0:3]) 
    labels = np.array(y_train[0:10])
    print(f"Testing on: \n{rows} \nLabels are: \n{labels}")
    best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = id3.find_best_split(rows, labels)

    print(f"Best gain {best_gain}")
    print(f"Best question {best_question}")
    print(f"True include {class_counts(best_true_rows, best_true_labels)}")
    print(f"False include {class_counts(best_false_rows, best_false_labels)}")


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
