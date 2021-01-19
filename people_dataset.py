import os
from random import randint
from random import sample
import cv2
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from utils import get_stratify


class PeopleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_ext, transform=None):
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform
        self.peopleId_list = os.listdir(img_dir)
        correct_id, error_ids = train_test_split(self.peopleId_list, test_size=0.2, random_state=41)
        self.id2label = {}
        for index in correct_id:
            self.id2label[int(index)] = 1
        for index in error_ids:
            self.id2label[int(index)] = 0
        self.groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11, 22], [12, 13, 14], [15, 16, 17], [18, 19], [20, 21]]

    def __len__(self):
        return len(self.peopleId_list)

    def __getitem__(self, idx):
        people_path = os.path.join(self.img_dir, str(idx + 1))
        people_meta = get_stratify(people_path)
        imgs = []
        classes = people_meta['class'].values.astype(int).tolist()
        imgs_trans = []
        for path in people_meta['path']:
            img = cv2.imread(path)
            imgs.append(img)

        if self.transform is not None:
            for img in imgs:
                augmented = self.transform(image=img)
                img_trans = augmented['image']
                imgs_trans.append(img_trans)

        if self.id2label[idx + 1] == 0:
            rand_group = randint(0, 6)
            error_group = self.groups[rand_group]
            rand_class = sample(error_group, 2)
            i = classes.index(rand_class[0])
            j = classes.index(rand_class[1])
            tmp = classes[i]
            classes[i] = classes[j]
            classes[j] = tmp
        return torch.stack(imgs_trans, 0), torch.tensor(classes), self.id2label[idx + 1], idx + 1