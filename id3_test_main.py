from ID3 import ID3
from utils import *


# File I created to test ID3

# TO RUN:
# python3 id3_test_main.py

def test_initiation(attributes_names):
    print("################################## test_initiation")
    id3 = ID3(attributes_names)

def test_entropy(attributes_names, x_train):
    print("################################## test_entropy")
    id3 = ID3(attributes_names)

    arr_perimeter_mean = np.array(x_train[:, 2])
    label_arr = np.where(arr_perimeter_mean <= 70, "Small", "Large" ) 
    # print(f"arr_perimeter_mean: min = {min(arr_perimeter_mean)} max = {max(arr_perimeter_mean)}")

    # Check entropy calc
    entropy = id3.entropy(arr_perimeter_mean, label_arr)
    print(f"Entropy = {entropy}")





if __name__ == '__main__':

    # Get data
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    target_attribute = 'diagnosis'
    (x_train, y_train, x_test, y_test) = get_dataset_split(train_dataset, test_dataset, target_attribute)

    print("-------------------------------------------")
    print("---------------- INIT DATA ----------------")
    print("-------------------------------------------")
    print("attributes_names:")
    print(attributes_names)
    print("x_train")
    print(x_train)
    print("y_train")
    print(y_train)
    print("x_test")
    print(x_test)
    print("y_test")
    print(y_test)
    print(f"Example in train.csv: Person 0: x_train[0] {x_train[0]}   [The data for Person 0]")
    print(f"Example in train.csv: Person 0: y_train[0] {y_train[0]}   [The diagnosis for Person 0]")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")


    # Tests
    print("-------------------------------------------")
    print("------------------ TESTS ------------------")
    print("-------------------------------------------")

    test_initiation(attributes_names)
    test_entropy(attributes_names, x_train)
