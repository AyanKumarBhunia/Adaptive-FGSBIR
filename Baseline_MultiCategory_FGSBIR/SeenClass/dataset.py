import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import numpy as np

unseen_classes = ['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon',
                  'rhinoceros', 'saw', 'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree', 'wheelchair', 'windmill', 'window']

import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################################
############## Train Case ##########################################
###################################################################
# single task for a class/ batch wise
# print(meta_batch['meta_batch']['meta_train']['sketch'].shape)
# print(meta_batch['meta_batch']['meta_train']['positive'].shape)
# print(meta_batch['meta_batch']['meta_train']['negative'].shape)
# print(meta_batch['meta_batch']['meta_train']['path'].shape)

# print(meta_batch['meta_batch']['meta_test']['sketch'].shape)
# print(meta_batch['meta_batch']['meta_test']['positive'].shape)
# print(meta_batch['meta_batch']['meta_test']['negative'].shape)
# print(meta_batch['meta_batch']['meta_test']['path'].shape)

# print(meta_batch['class_label'])

###################################################################
############## Test Case ##########################################
###################################################################
#  There is no batch concept --> sample evaluation case for a particual class
# batch['meta_batch']['meta_train']['sketch']
# batch['meta_batch']['meta_train']['positive']
# batch['meta_batch']['meta_train']['path']

# batch['meta_batch']['meta_test']['sketch']
# batch['meta_batch']['meta_test']['positive']
# batch['meta_batch']['meta_test']['positive_path'] all 210 photos from 21 unseen class
# batch['meta_batch']['meta_test']['sketch_path'] --> all sketch under a particual class
# batch['class_name']


class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        with open('./../../Dataset/sketchy_all.pickle', 'rb') as fp:
            train_sketch, test_sketch, self.negetiveSampleDict, self.Coordinate = pickle.load(fp)

        # self.Train_Sketch, self.classWise_PhotoList, self.negetiveSampleDict
        # self.Train_metatrain_Sketch, self.Test_metatrain_Sketch, self.Test_metatest_Sketch
        # self.Train_Class,  self.Test_Class, self.all_class

        all_class = list(set([x.split('/')[0] for x in train_sketch]))
        all_class.sort()
        class_photo_list = {}

        self.Train_metatrain_Sketch, self.Test_metatrain_Sketch, self.Test_metatest_Sketch  = {}, {}, {}

        for x in all_class:
            class_photo_list[x], self.Train_metatrain_Sketch[x], \
            self.Test_metatrain_Sketch[x], self.Test_metatest_Sketch[x]  = [], [], [], []

        for x in train_sketch:
            class_photo_list[x.split('/')[0]].append(x.split('/')[0] + '/' + x.split('/')[1].split('-')[0])

        for x in class_photo_list.keys():
            class_photo_list[x] = list(set(class_photo_list[x]))
            test_id = random.sample(range(len(class_photo_list[x])), 10)
            train_id = list(set(range(len(class_photo_list[x]))) - set(test_id))

            sketch_list = np.array(class_photo_list[x])[train_id]
            self.Train_metatrain_Sketch[x] = [y for y in train_sketch if y.split('/')[0] + '/' + y.split('/')[1].split('-')[0] in sketch_list]

            sketch_list = np.array(class_photo_list[x])[test_id]
            self.Test_metatrain_Sketch[x] = [y for y in train_sketch if y.split('/')[0] + '/' + y.split('/')[1].split('-')[0] in sketch_list]


        self.all_class = all_class
        self.Train_Class = all_class
        self.Test_Class = all_class

        self.negetiveSampleDict = class_photo_list

        self.classWise_PhotoList = {}
        for x in all_class:
            self.classWise_PhotoList[x] = []
        for x in train_sketch + test_sketch:
            self.classWise_PhotoList[x.split('/')[0]].append(x.split('/')[0] + '/' + x.split('/')[1].split('-')[0])
        for x in self.classWise_PhotoList.keys():
            self.classWise_PhotoList[x] = list(set(self.classWise_PhotoList[x]))

        for x in test_sketch:
            self.Test_metatest_Sketch[x.split('/')[0]].append(x)

        self.Train_Sketch = []
        for x in self.Train_metatrain_Sketch.keys():
            self.Train_Sketch.extend(self.Train_metatrain_Sketch[x])

        np.random.shuffle(self.Train_Sketch)

        key = [x for x in self.Train_Class]
        value = [x for x in range(len(self.Train_Class))]
        self.string2label = {}
        self.label2string = {}
        for i in range(len(key)):
            self.string2label[key[i]] = value[i]
            self.label2string[value[i]] = key[i]


        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')

        self.hp = hp
        self.hp.shot = 5
        self.mode = mode


    def __getitem__(self, item):


        sample  = {}
        if self.mode == 'Train':

            sketch_path = self.Train_Sketch[item]
            positive_sample = self.Train_Sketch[item].split('/')[0] + '/' + self.Train_Sketch[item].split('/')[1].split('-')[0]
            positive_path = os.path.join(self.hp.root_dir, 'photo', positive_sample + '.jpg')

            class_name = self.Train_Sketch[item].split('/')[0]

            possible_list = self.negetiveSampleDict[class_name].copy()
            possible_list.remove(positive_sample)
            negative_sample = possible_list[randint(0, len(possible_list) - 1)]
            negative_path = os.path.join(self.hp.root_dir, 'photo', negative_sample + '.jpg')

            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch': sketch_img, 'sketch_path': sketch_path,
                      'positive': positive_img, 'positive_path': positive_sample,
                      'negative': negative_img, 'negative_path': negative_sample,
                      'class_label': self.string2label[class_name]
                      }


        elif self.mode == 'Test':

            batch = {'meta_train': {}, 'meta_test': {}}

            meta_train_sketch = self.Test_metatrain_Sketch[self.Test_Class[item]]
            meta_train_ind = random.sample(range(len(meta_train_sketch)), self.hp.shot)
            meta_train_sketch = np.array(meta_train_sketch)[meta_train_ind]

            sketch_buffer, positive_buffer, negative_buffer, path_buffer = [], [], [], []

            for sketch_path in meta_train_sketch:

                path_buffer.append(sketch_path)
                positive_sample = sketch_path.split('/')[0] + '/' + \
                                  sketch_path.split('/')[1].split('-')[0]
                positive_path = os.path.join(self.hp.root_dir, 'photo', positive_sample + '.jpg')

                possible_list = self.negetiveSampleDict[self.Test_Class[item]].copy()
                possible_list.remove(positive_sample)

                # try:
                #     possible_list.remove(positive_sample)
                # except:
                #     print(positive_sample)

                negative_sample = possible_list[randint(0, len(possible_list) - 1)]
                negative_path = os.path.join(self.hp.root_dir, 'photo', negative_sample + '.jpg')

                vector_x = self.Coordinate[sketch_path]
                sketch_img = rasterize_Sketch(vector_x)
                sketch_img = Image.fromarray(sketch_img).convert('RGB')

                positive_img = Image.open(positive_path).convert('RGB')
                negative_img = Image.open(negative_path).convert('RGB')

                sketch_img = self.train_transform(sketch_img)
                positive_img = self.train_transform(positive_img)
                negative_img = self.train_transform(negative_img)

                sketch_buffer.append(sketch_img)
                positive_buffer.append(positive_img)
                negative_buffer.append(negative_img)

            batch['meta_train']['sketch'] = torch.stack(sketch_buffer, dim=0)
            batch['meta_train']['positive'] = torch.stack(positive_buffer, dim=0)
            batch['meta_train']['negative'] = torch.stack(negative_buffer, dim=0)
            batch['meta_train']['path'] = path_buffer

            meta_test_sketch = self.Test_metatest_Sketch[self.Test_Class[item]]
            sketch_buffer, sketch_path_buffer = [], []

            for sketch_path in meta_test_sketch:
                vector_x = self.Coordinate[sketch_path]
                sketch_img = rasterize_Sketch(vector_x)
                sketch_img = Image.fromarray(sketch_img).convert('RGB')

                sketch_img = self.train_transform(sketch_img)
                sketch_buffer.append(sketch_img)
                sketch_path_buffer.append(sketch_path)

            positive_buffer, positive_path_buffer = [], []

            for positive_sample in self.classWise_PhotoList[self.Test_Class[item]]:

                positive_path = os.path.join(self.hp.root_dir, 'photo', positive_sample + '.jpg')
                positive_img = Image.open(positive_path).convert('RGB')
                positive_img = self.train_transform(positive_img)

                positive_buffer.append(positive_img)
                positive_path_buffer.append(positive_path)


            batch['meta_test']['sketch'] = torch.stack(sketch_buffer, dim=0)
            batch['meta_test']['positive'] = torch.stack(positive_buffer, dim=0)
            batch['meta_test']['positive_path'] = positive_path_buffer
            batch['meta_test']['sketch_path'] = sketch_path_buffer

            sample = {'meta_batch': batch, 'class_name': self.Test_Class[item]}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            # return len(self.Test_metatest_Sketch[self.currentTestClass])
            return len(self.Test_Class)

def collate_self(batch):
    return batch

def get_dataloader(hp):

    dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads))

    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    # batch = dataset_Test[0]
    # for x in dataset_Test:
    #     print(x)


    return dataloader_Train, dataloader_Test

def get_ransform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(299)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

