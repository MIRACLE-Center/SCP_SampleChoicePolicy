import itertools
from scipy.spatial.distance import cdist
import torchvision.transforms as transforms
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
from augment import cc_augment
import csv
import torchvision.transforms.functional as TF


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    # image.save('raw.jpg')
    aug_image = aug_transform(image)
    # aug_image_PIL = to_PIL(aug_image)
    # aug_image_PIL.save('aug.jpg')
    # import ipdb; ipdb.set_trace()
    return aug_image


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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


# '/home/quanquan/hand/hand/histo-norm/'
class HandXray(data.Dataset):
    def __init__(self, pathDataset='/home/quanquan/hand/hand/jpg/', label_path='/home/quanquan/hand/hand/all.csv', \
                 mode="Oneshot", size=[384, 384], extra_aug_loss=False, patch_size=192, rand_psize=False,
                 retfunc=2):
        """
        extra_aug_loss: return an additional image for image augmentation consistence loss
        """
        self.size = size
        self.extra_aug_loss = extra_aug_loss  # consistency loss
        self.pth_Image = os.path.join(pathDataset)
        # self.patch_scale = patch_scale
        self.patch_size = patch_size
        if rand_psize:
            print("Using Random patch Size")
            self.rand_psize = rand_psize
            self.patch_size = -1

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

        # for i in range(start, end + 1):
        #     self.list.append({'ID': "{0:03d}".format(i)})

        # num_repeat = 10
        # if mode == 'Train':
        #     temp = self.list.copy()
        #     for _ in range(num_repeat):
        #         self.list.extend(temp)

        self.retfunc = retfunc
        self.mode = mode
        self.base = 16

    def __getitem__(self, index):
        if self.retfunc == 2:
            return self.retfunc2(index)
        elif self.retfunc == 3:
            return self.retfunc3(index)
        elif self.retfunc == 1:
            return self.retfunc1(index)
        elif self.retfunc == 4:
            return self.retfunc4(index)

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 10) * 32
        print("[debug] : patch-size set as ", self.patch_size)

    def retfunc1(self, index):
        """
        Original Point Choosing Function
        """
        np.random.seed()
        item = dict()

        if self.transform != None:
            img_pil = Image.open(self.train_list[index]).convert('RGB')
            item['image'] = self.transform(img_pil)  # typeerror: 'numpy.str_' object does not support item assignment ?
            if self.extra_aug_loss:
                item['extra_image'] = self.extra_aug_transform(img_pil)
            # print("[debug] ", item['image'].shape, item['extra_image'].shape)

        # Crop 192 x 192 Patch
        # patch_size = int(self.patch_scale * self.size[0])
        patch_size = self.patch_size
        margin_x = np.random.randint(0, self.size[0] - patch_size)
        margin_y = np.random.randint(0, self.size[0] - patch_size)
        crop_imgs = augment_patch(item['image'] \
                                      [:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size], \
                                  self.aug_transform)

        chosen_x = np.random.randint(int(0.1 * patch_size), int(0.9 * patch_size))
        chosen_y = np.random.randint(int(0.1 * patch_size), int(0.9 * patch_size))
        raw_y, raw_x = chosen_y + margin_y, chosen_x + margin_x

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y, chosen_x] = 1
        # print("[debug2s] ", crop_imgs.shape, temp.shape)
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        # to_PIL(crop_imgs).save('img_after.jpg')
        temp = temp[3]
        # print(chosen_y, chosen_x)
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size
        # print(chosen_y, chosen_x)
        # import ipdb; ipdb.set_trace()
        if self.extra_aug_loss:
            return item['image'], crop_imgs, chosen_y, chosen_x, raw_y, raw_x, item['extra_image']
        return item['image'], crop_imgs, chosen_y, chosen_x, raw_y, raw_x

    def retfunc2(self, index):
        """
        New Point Choosing Function
        """
        np.random.seed()
        item = dict()
        if self.transform != None:
            img_pil = Image.open(self.train_list[index]).convert('RGB')
            item['image'] = self.transform(img_pil)  # typeerror: 'numpy.str_' object does not support item assignment ?

        padding = int(0.1*self.size[0])
        patch_size = self.patch_size
        raw_x, raw_y = self.select_point_from_prob_map(self.prob_map, size=self.size)
        while True:
            left = np.random.randint(0, min(raw_x, self.size[0]-patch_size))
            if raw_x - left <= patch_size - padding:
                break
        while True:
            top = np.random.randint(0, min(raw_y, self.size[0]-patch_size))
            if raw_y - top <= patch_size - padding:
                break
        margin_x = left
        margin_y = top
        # print("margin x y", margin_y, margin_x, patch_size)
        # import ipdb; ipdb.set_trace()
        cimg = item['image'][:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_x, chosen_y = raw_x - margin_x, raw_y - margin_y

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y, chosen_x] = 1
        # print("[debug2s] ", crop_imgs.shape, temp.shape)
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        # to_PIL(crop_imgs).save('img_after.jpg')
        temp = temp[3]
        # print(chosen_y, chosen_x)
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size
        # print(chosen_y, chosen_x)
        # import ipdb; ipdb.set_trace()
        return item['image'], crop_imgs, chosen_y, chosen_x, raw_y, raw_x

    def retfunc3(self, index):
        """
        Two Crops Consistency
        """
        padsize = (32,32)
        np.random.seed()
        item = dict()
        if self.transform != None:
            img_pil = Image.open(self.train_list[index]).convert('RGB')
            item['image'] = self.transform_resize(img_pil)  # typeerror: 'numpy.str_' object does not support item assignment ?

        p0, p1, (x0, x1, y0, y1) = self.select_dual_points(self.size, (32,32))
        crop_img1 = TF.crop(item['image'], *p0, *padsize)
        crop_img2 = TF.crop(item['image'], *p1, *padsize)

        # TODO:
        # augment(crop_img1, crop_img2)
        crop_img1 = self.transform_tensor(crop_img1)
        crop_img2 = self.transform_tensor(crop_img2)
        return crop_img1, crop_img2, p0, p1, (x0, x1, y0, y1)

    def retfunc4(self, index):
        """
        Only for Testing
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        item_path = self.train_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape[::-1]  # shape: (y, x) or (long, width)
        # if self.transform is not None:
        item = self.transform(ori_img.convert('RGB'))
        # import ipdb; ipdb.set_trace()

        landmark = self.resize_landmark(self.landmarks_train[index]['landmark'],
                                        img_shape)  # [1:] for discarding the index
        return item, landmark, img_shape

    def get_points_and_masks(self, imsize=(256,256), padsize=(32,32)):
        """
        Arbitarily Get two points and their corresponding overlapping masks
        """
        x0 = np.random.randint(0,  imsize[0]-padsize[0])
        x1 = np.random.randint(x0, imsize[0]-padsize[0])
        y0 = np.random.randint(0,  imsize[1]-padsize[1])
        y1 = np.random.randint(y0, imsize[1]-padsize[1])
        # left point: left_top or left_bottom
        if np.random.rand() > 0.5:
            p0 = (x0, y0) # x0 y0 < x1, y1
            p1 = (x1, y1)
            mask0 = np.zeros(padsize)
            mask1 = np.zeros(padsize)
            mask0[x1-x0:padsize[0], y1-y0:padsize[1]] = 1
            mask1[x0-x1+padsize[0]:padsize[0], y0-y1+padsize[1]:padsize[1]] = 1
        else:
            p0 = (x0, y1)
            p1 = (x1, y0)
            mask0[x1-x0:padsize[0], y1-y0:padsize[1]] = 1
            mask1[x0-x1+padsize[0]:padsize[0], y0-y1+padsize[1]:padsize[1]] = 1
        return p0, p1, (x0, x1, y0, y1)

    def __len__(self):
        if self.istrain:
            return len(self.train_list)
        else:
            return len(self.test_list)


class HandXrayLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(HandXrayLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_rand_psize()
        return super(HandXrayLoader, self).__iter__()


class TestHandXray(data.Dataset):
    def __init__(self, pathDataset='/home/quanquan/hand/hand/jpg/', label_path='/home/quanquan/hand/hand/all.csv',
                 mode=None, istrain=False, size=[384, 384], load_mod="img"):
        self.istrain = istrain
        self.size = size
        self.original_size = [1, 1]
        self.label_path = label_path
        self.num_landmark = 37  # for hand example

        self.pth_Image = os.path.join(pathDataset)

        self.list = np.array([x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")])
        self.list.sort()
        # print(self.list)
        self.test_list = self.list[:20]
        self.train_list = self.list[20:]
        self.landmarks = get_csv_content(label_path)
        # for landmark in self.landmarks:
        #     print("landmark indx", landmark['index'])
        self.landmarks_test = self.landmarks[:20]
        self.landmarks_train = self.landmarks[20:]
        print("train_list: ", len(self.train_list))
        print("test_list : ", len(self.test_list))

        # Read all imgs to mem in advanced
        # self.train_items = [Image.open(x) for x in self.train_list]
        # self.test_items = [Image.open(x) for x in self.test_list]

        # transform for both test/train
        normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.simple_trans = transforms.Compose(transformList[1:])

        # transform for only train
        transform_list = [
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        self.aug_transform = transforms.Compose(transform_list)

        # self.maxHammingSet = generate_hamming_set(9, 100)
        self.mode = mode

    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(x,y), (x,y), (x,y), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[0] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[1] / float(img_shape[1]))
        return landmark
        # for i in range(len(landmark)):
        #     landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        # return landmark

    def _get_original(self, index):
        item_path = self.test_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape
        item = self.simple_trans(ori_img.convert('RGB'))
        return item, self.landmarks_test[index]['landmark'],

    def get_oneshot_name(self, index):
        img_path = self.train_list[index]
        parent, name = os.path.split(img_path)
        return parent, name

    def get_one_shot(self, index=1, debug=False):  # some bugs in index=0
        img_path = self.train_list[index]
        print("oneshot img pth: ", img_path)
        ori_img = Image.open(img_path)
        # if self.transform is not None:
        img = self.transform(ori_img.convert('RGB'))
        shape = np.array(ori_img).shape[::-1]
        landmark_list = self.resize_landmark(self.landmarks_train[index]['landmark'], shape)
        if debug:  # The part above is correct!
            return img, landmark_list, shape  # disacrd the index

        ##############################################
        template_patches = torch.zeros([self.num_landmark, 3, 192, 192])
        for id, landmark in enumerate(landmark_list):
            left = min(max(landmark[0] - 96, 0), 192)
            bottom = min(max(landmark[1] - 96, 0), 192)
            template_patches[id] = img[:, bottom:bottom + 192, left:left + 192]
            landmark_list[id] = [landmark[0] - left, landmark[1] - bottom]
            # if id == 9:
            #     print(landmark)
            #     print(left, bottom)
            #     to_PIL(template_patches[id]).save('template.jpg')
        return img, landmark_list, template_patches

    def __getitem__(self, index):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        if self.istrain:
            item_path = self.train_list[index]
        else:
            item_path = self.test_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape[::-1]  # shape: (y, x) or (long, width)
        # if self.transform is not None:
        item = self.transform(ori_img.convert('RGB'))
        # import ipdb; ipdb.set_trace()

        if self.istrain:
            landmarks = self.landmarks_train[index]['landmark']
        else:
            landmarks = self.landmarks_test[index]['landmark']
        landmark = self.resize_landmark(landmarks, img_shape)  # [1:] for discarding the index
        assert landmark is not None, f"Got Landmarks None, {item_path}"
        assert img_shape is not None, f"Got Landmarks None, {item_path}"
        return item, landmark, img_shape

    def __len__(self):
        if self.istrain:
            return len(self.train_list)
        else:
            return len(self.test_list)


def histo_normalize(img, ref, name=None):
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):
        # print(i)
        hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx
    if name is not None:
        cv2.imwrite(f'/home/quanquan/hand/hand/histo-norm/{name}', out)
        print(f'Save: {name}')


def app_histo_norm():
    # from tqdm import tqdm
    data_root = "/home/quanquan/hand/hand/jpg/"
    ref_path = data_root + "3143.jpg"
    ref = cv2.imread(ref_path)
    for x in os.scandir(data_root):
        if x.name.endswith(".jpg"):
            img = cv2.imread(x.path)
            histo_normalize(img, ref, name=x.name)


def test_HandXray_img():
    from utils import visualize
    from tqdm import tqdm
    # hamming_set(9, 100)
    test = HandXray(patch_size=208)
    for i in tqdm(range(2, 3)):
        item, crop_imgs, chosen_y, chosen_x, raw_y, raw_x = test.__getitem__(i)
        vis1 = visualize(item.unsqueeze(0), [[raw_x, raw_y]], [[raw_x, raw_y]])
        vis1.save(f"imgshow/train_{i}.jpg")
        vis2 = visualize(crop_imgs.unsqueeze(0), [[chosen_x, chosen_y]], [[chosen_x, chosen_y]])
        vis2.save(f"imgshow/train_{i}_crop.jpg")
        print("logging ", item.shape)
        print("crop", crop_imgs.shape)
        # for i in range(9):
        #     reimg = inv_trans(crop_imgs_1[:,:,:,i].transpose((1,2,0))).numpy().transpose((1,2,0))*255
        #     print(reimg.shape)
        #     cv2.imwrite(f"tmp/reimg_{i}.jpg", reimg.astype(np.uint8))
        import ipdb;
        ipdb.set_trace()
    print("pass")


def test3():
    from utils import visualize
    test = TestHandXray()
    img, landmark_list, shape = test.get_one_shot(1, True)
    print(landmark_list)
    vis = visualize(img.unsqueeze(0), landmark_list, landmark_list)
    vis.save(f'imgshow/refer2_{1}.jpg')
    print("dasdas")


def test4():
    from utils import visualize
    test = TestHandXray()
    dataloader_1 = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)
    for img, landmark, shape in dataloader_1:
        vis = visualize(img, landmark, landmark)
        vis.save(f'imgshow/dataloader_{1}.jpg')
        print("save")


if __name__ == "__main__":
    # hamming_set(9, 100)
    test_HandXray_img()
    # test2()
    # test4()
    # app_histo_norm()
