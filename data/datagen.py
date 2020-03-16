import os
import cv2
import sys
import glob
import pickle
import random
import tqdm
import numpy as np

TEST_PERCLASS_NUM = 100
def generate_hand_dataset(path):
    dirs = os.listdir(path)
    train_data_list = [[] for _ in range(len(dirs))]
    test_data_list = [[] for _ in range(len(dirs))]
    for i, dir in enumerate(dirs):
        files = glob.glob(os.path.join(path, os.path.join(dir, "*.*")))
        random.shuffle(files)
        subdata = zip(files, [i]*len(files))
        for j, (file, label) in enumerate(subdata):
            if j < TEST_PERCLASS_NUM:
                test_data_list[i].append((file, label))
            else:
                train_data_list[i].append((file, label))

    train_data = []
    test_data = []
    max = 0
    for i in range(len(train_data_list)):
        if max < len(train_data_list[i]):
            max = len(train_data_list[i])
    print("max number : {}".format(max))
    for i in range(len(train_data_list)):
        sublen = len(train_data_list[i])
        print("{} repeats {} files".format(dirs[i], max - sublen))
        while max - sublen > sublen:
            train_data_list[i].extend(train_data_list[i][:])
            sublen = len(train_data_list[i])
        train_data_list[i].extend(train_data_list[i][:max - sublen])
        train_data.extend(train_data_list[i])

    for i in range(len(test_data_list)):
        test_data.extend(test_data_list[i])

    random.shuffle(train_data)
    random.shuffle(test_data)
    print("total train data : {}".format(len(train_data)))
    print("total test data : {}".format(len(test_data)))

    train_lines = ""
    for file, label in train_data:
        train_lines += file + " " + str(label) + "\n"
    test_lines = ""
    for file, label in test_data:
        test_lines += file + " " + str(label) + "\n"
    with open("train.txt", "w") as train_f:
        train_f.writelines(train_lines)
    with open("test.txt", "w") as test_f:
        test_f.writelines(test_lines)

if __name__ == '__main__':
    hand_path = sys.argv[1]
    generate_hand_dataset(hand_path)
