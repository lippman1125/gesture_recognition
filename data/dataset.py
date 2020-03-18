import os
import sys
import torch
import torch.utils.data as data
import cv2
import tqdm
from PIL import Image
import numpy as np

class hand_dataset(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, anno, transform=None, preload = True):
        self.root = root
        self.anno = anno
        self.transform = transform
        self.ids = list()
        with open(os.path.join(root, anno), 'r') as f:
          self.ids = [tuple(line.strip('\n').split()) for line in f]

        self.preload = preload
        self.preload_data = []
        if preload:
            for i in tqdm.tqdm(range(len(self.ids))):
                img_id = self.ids[i]
                path, label = img_id
                img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.preload_data.append((img, label))
            print("{} Data load completed".format(anno))

    def __getitem__(self, index):
        if self.preload:
            img, label = self.preload_data[index]
        else:
            img_id = self.ids[index]
            path, label = img_id
            img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (64,64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.open(os.path.join(self.root, path))
            # img_resized = img.resize((32, 32))

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        label = int(label)

        return img, label

    def get_mean_std(self):
        r_mean, g_mean, b_mean = 0.0, 0.0, 0.0
        r_std, g_std, b_std = 0.0, 0.0, 0.0
        num = len(self.preload_data)
        if self.preload:
            for i in range(num):
                # print(np.shape(self.preload_data[i][0]))
                r_mean += np.mean(self.preload_data[i][0][:,:,0])/255.0
                g_mean += np.mean(self.preload_data[i][0][:,:,1])/255.0
                b_mean += np.mean(self.preload_data[i][0][:,:,2])/255.0
                r_std += np.std(self.preload_data[i][0][:,:,0])/255.0
                g_std += np.std(self.preload_data[i][0][:,:,1])/255.0
                b_std += np.std(self.preload_data[i][0][:,:,2])/255.0

            r_mean, g_mean, b_mean = round(r_mean/num, 3), round(g_mean/num, 3), round(b_mean/num, 3)
            r_std, g_std, b_std = round(r_std/num, 3), round(g_std/num, 3), round(b_std/num, 3)
            print("RGB Mean = {} {} {}".format(r_mean, g_mean, b_mean))
            print("RGB Std = {} {} {}".format(r_std, g_std, b_std))



    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    class_dict = {0:"gun", 1:"thumbup", 2:"victory", 3:"negative", 4:"ok"}
    train_data = hand_dataset("./", "test.txt")
    for img, label in train_data:
        cv2.imshow("hand disp", img[:,:,::-1])
        print(class_dict[label])
        cv2.waitKey(0)

