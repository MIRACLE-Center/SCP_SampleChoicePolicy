import torchvision.transforms as transforms
import numpy as np
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
import csv
import json
from tutils import tfilename, tdir



def get_csv_content(path):  #
    if not path.endswith('.csv'):
        raise ValueError(f"Wrong path, Got {path}")
    with open(path) as f:
        f_csv = csv.reader(f)
        res_list = []
        discard = next(f_csv)  # discard No.3142
        for i, row in enumerate(f_csv):
            res = {'index': row[0]}
            landmarks = []
            for i in range(1, len(row), 2):
                # print(f"{i}")
                # landmarks += [(row[i], row[i+1])] #
                landmarks += [[int(row[i]), int(row[i + 1])]]  # change _tuple to _list
            res['landmark'] = landmarks
            res_list += [res]
        return res_list

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class HandXray(data.Dataset):
    def __init__(self,
                 pathDataset='/home/quanquan/hand/hand/jpg/',
                 label_path='/home/quanquan/hand/hand/all.csv',
                 mode="Oneshot",
                 size=[384, 384],
                 R_ratio=0.05,
                 num_landmark=37,
                 pseudo_pth=None):
        self.pseudo_pth = pseudo_pth
        self.num_landmark = num_landmark
        self.size = size
        self.pth_Image = os.path.join(pathDataset)

        self.list = [x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")]
        self.list.sort()
        # print(self.list)
        self.landmarks = get_csv_content(label_path)
        self.test_list = self.list[:200]
        self.landmarks_test = self.landmarks[:200]
        self.train_list = self.list[200:]
        self.landmarks_train = self.landmarks[200:]

        if mode in ["Oneshot", "Train"]:
            self.istrain = True
        elif mode in ["Test1", "Test"]:
            self.istrain = False
        else:
            raise NotImplementedError

        normalize = transforms.Normalize([0.], [1.])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.transform_resize = transforms.Resize(self.size)
        self.transform_tensor = transforms.ToTensor()

        # transforms.RandomChoice(transforms)
        # transforms.RandomApply(transforms, p=0.5)
        # transforms.RandomOrder()
        self.extra_aug_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.25)], p=0.5),
            transforms.ToTensor(),
            AddGaussianNoise(0., 1.),
            transforms.Normalize([0], [1]),
        ])

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        )
        self.mode = mode
        self.base = 16

        # gen mask
        self.Radius = int(max(size) * R_ratio)
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1

        self.mask = mask
        self.guassian_mask = guassian_mask

        # gen offset
        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

    def __getitem__(self, index):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        item_path = self.train_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape[::-1]  # shape: (y, x) or (long, width)
        # if self.transform is not None:
        item = self.transform(ori_img.convert('RGB'))
        # import ipdb; ipdb.set_trace()

        if self.mode != 'pseudo':
            landmark_list = self.resize_landmark(self.landmarks_train[index]['landmark'],
                                                 img_shape)  # [1:] for discarding the index
        else:
            p, name = os.path.split(item_path)
            landmark_list = []
            with open(tfilename(self.pseudo_pth, f"{name[:-4]}.json"), 'r') as f:
                landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)

        y, x = item.shape[-2], item.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        return {'img':item, 'mask':mask, 'offset_x': offset_x, 'offset_y':offset_y, 'landmark_list': landmark_list, "img_shape": img_shape}


    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(x,y), (x,y), (x,y), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[0] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[1] / float(img_shape[1]))
        return landmark

    def __len__(self):
        if self.istrain:
            return len(self.train_list)
        else:
            return len(self.test_list)


def TestHandXray(*args, **kwargs):
    return HandXray(mode="Test", *args, **kwargs)


if __name__ == '__main__':
    dataset = HandXray(pathDataset='/home1/quanquan/datasets/hand/hand/jpg/', label_path='/home1/quanquan/datasets/hand/hand/all.csv')
    img = dataset.__getitem__(0)
    import ipdb; ipdb.set_trace()