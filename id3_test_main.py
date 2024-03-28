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

def test_entropy(attributes_names, y_train):
    print("################################## test_entropy")
    id3 = ID3(attributes_names)

    # We'll check the entropy of the root node, meaning the node with the entirety of y_train
    rows_arr = np.array(y_train)
    label_arr = np.array(y_train)

    counts = class_counts(rows_arr, label_arr)
    print(f"counts = {counts}")

    # Check entropy calc
    entropy = id3.entropy(rows_arr, label_arr)
    print(f"Entropy = {entropy}")

def test_information_gain_aux(attributes_names, y_train, full_range, left_side, right_side):
    id3 = ID3(attributes_names)

    # We'll check the IG of perfect split
    full_rows_arr = np.array(y_train[full_range[0]:full_range[1]])
    full_label_arr = np.array(y_train[full_range[0]:full_range[1]])
    L_rows_arr = np.array(y_train[left_side[0]:left_side[1]])
    L_label_arr = np.array(y_train[left_side[0]:left_side[1]])
    R_rows_arr = np.array(y_train[right_side[0]:right_side[1]])
    R_label_arr = np.array(y_train[right_side[0]:right_side[1]])

    parent_node_entropy = id3.entropy(full_rows_arr, full_label_arr)

    print(helper_print_counts_and_entropy("Full",full_rows_arr, full_label_arr, id3))
    print(helper_print_counts_and_entropy("Left",L_rows_arr, L_label_arr, id3))
    print(helper_print_counts_and_entropy("Right",R_rows_arr, R_label_arr, id3))

    info_gain = id3.info_gain(L_rows_arr, L_label_arr, R_rows_arr, R_label_arr, parent_node_entropy)
    print(f"Info Gain = {info_gain}")

def test_information_gain_perfect_split(attributes_names, y_train):
    print("################################## test_information_gain_perfect_split")
    test_information_gain_aux(attributes_names, y_train, (0,6), (0,2), (3,6))

def test_information_gain_good_split(attributes_names, y_train):
    print("################################## test_information_gain_good_split")
    test_information_gain_aux(attributes_names, y_train, (0,6), (0,3), (4,6))
    




if __name__ == '__main__':

    # Get data
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    target_attribute = 'diagnosis'
    (x_train, y_train, x_test, y_test) = get_dataset_split(train_dataset, test_dataset, target_attribute)

    # print("-------------------------------------------")
    # print("---------------- INIT DATA ----------------")
    # print("-------------------------------------------")
    # print("attributes_names:")
    # print(attributes_names)
    # print("x_train")
    # print(x_train)
    # print("y_train")
    # print(y_train)
    # print("x_test")
    # print(x_test)
    # print("y_test")
    # print(y_test)
    # print(f"Example in train.csv: Person 0: x_train[0] {x_train[0]}   [The data for Person 0]")
    # print(f"Example in train.csv: Person 0: y_train[0] {y_train[0]}   [The diagnosis for Person 0]")
    # print("-------------------------------------------")
    # print("-------------------------------------------")
    # print("-------------------------------------------")


    # Tests
    print("-------------------------------------------")
    print("------------------ TESTS ------------------")
    print("-------------------------------------------")

    test_initiation(attributes_names)
    test_entropy(attributes_names, y_train)
    test_information_gain_perfect_split(attributes_names, y_train)
    test_information_gain_good_split(attributes_names, y_train)
