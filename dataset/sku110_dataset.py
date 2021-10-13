import os
import glob
import torch
import csv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as FT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import ImageFile
from scipy.stats import multivariate_normal

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize(image, boxes, dims_800=True, return_percent_coords=False):
    # torch to PIL
    image = FT.to_pil_image(image)

    if not dims_800:
        w, h = image.size
        wh = np.array([w, h])
        smallest_side = np.min(wh)

        min_idx = np.argmin(wh)
        max_idx = np.argmax(wh)

        min_scale = 800 / smallest_side
        largest_side = np.max(wh)
        if min_scale * largest_side <= 1333:
            max_scale = min_scale
        else:
            max_scale = 1333 / largest_side

        if min_idx == 0:
            new_w = int(wh[min_idx] * min_scale)
            new_h = int(wh[max_idx] * max_scale)
        elif min_idx == 1:
            new_w = int(wh[max_idx] * max_scale)
            new_h = int(wh[min_idx] * min_scale)

        if new_w < 800:
            new_w = 800
        if new_w > 1333:
            new_w = 1333

        if new_h < 800:
            new_h = 800
        if new_h > 1333:
            new_h = 1333

        dims = (new_h, new_w)
    else:
        dims = (800, 800)

    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes, dims


class SKU110K_Dataset(Dataset):
    def __init__(self, root, split, resize=800, visualize=False):
        super().__init__()
        assert split in ['train', 'val', 'test']

        self.root = root
        self.split = split
        self.resize = resize
        self.root = os.path.join(self.root, 'SKU110K_fixed')
        self.images_path_ = os.path.join(self.root, 'images')
        self.images_path = glob.glob(os.path.join(self.images_path_, '{}_*.jpg').format(self.split))
        self.annotations_path = os.path.join(self.root, 'annotations')
        self.annotations_path = os.path.join(self.annotations_path, 'annotations_{}.csv'.format(self.split))
        self.img_total_num = 0

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.visualize = visualize

        self.bbox_dict = {}
        with open(self.annotations_path, mode='r') as f:
            reader = csv.reader(f)
            for line in reader:
                img_name = line[0]
                bbox = list(map(int, line[1:5]))
                wh = list(map(int, line[-2:]))
                if img_name not in self.bbox_dict:
                    self.bbox_dict[img_name] = []
                    self.bbox_dict[img_name].append(bbox)
                else:
                    self.bbox_dict[img_name].append(bbox)
        self.img_total_num = len(self.bbox_dict)
        print("num of {} image :{}".format(self.split, self.img_total_num))

        # remove data
        if self.split == 'train':
            # During the training time, we excluded erroneous data that could cause the loss explosion.
            except_img_name = ['train_1465.jpg',  # nan
                               'train_1718.jpg', 'train_2883.jpg', 'train_3029.jpg', 'train_3220.jpg',
                               'train_3232.jpg', 'train_3580.jpg', 'train_3760.jpg', 'train_3774.jpg', 'train_5418.jpg',
                               'train_5602.jpg', 'train_6381.jpg', 'train_6474.jpg', 'train_6566.jpg', 'train_7099.jpg',
                               'train_7895.jpg',  # 16
                               'train_1121.jpg', 'train_1713.jpg', 'train_2820.jpg', 'train_3252.jpg', 'train_3622.jpg',
                               'train_4668.jpg', 'train_4626.jpg', 'train_4741.jpg', 'train_5318.jpg', 'train_5440.jpg',
                               'train_5822.jpg', 'train_5966.jpg', 'train_6216.jpg', 'train_8046.jpg', 'train_8082.jpg',
                               'train_939.jpg',   # 16
                               'train_3832.jpg'   # wrong data
                                ]

            for except_name in except_img_name:
                del self.images_path[self.images_path.index(os.path.join(self.images_path_, except_name))]

        if self.split == 'test':
            # erroneous data, see issue.
            except_img_name = ['test_22.jpg',
                               'test_132.jpg',
                               'test_184.jpg',
                               'test_232.jpg',
                               'test_910.jpg',
                               'test_923.jpg',
                               'test_1346.jpg',
                               'test_2028.jpg',
                               'test_2321.jpg']

            for except_name in except_img_name:
                del self.images_path[self.images_path.index(os.path.join(self.images_path_, except_name))]

    def __len__(self):
        return len(self.images_path)

    def visualize_box(self, image, boxes, wh, denormalize=True, gaussian_map=None):
        # tensor to img
        if denormalize:
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis *= self.std
            img_vis += self.mean
            img_vis = np.clip(img_vis, 0, 1)
        else:
            img_vis = image
        plt.figure('input')
        plt.imshow(img_vis)

        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
        cycx = torch.cat([cxcy[:, 1:2], cxcy[:, 0:1]], dim=1)
        boxes *= wh

        for i in range(len(boxes)):
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]

            plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                          width=x2 - x1,
                                          height=y2 - y1,
                                          linewidth=1,
                                          edgecolor=(1, 0, 1),
                                          facecolor='none'))

            cx = cxcy[i][0]  # y
            cy = cxcy[i][1]  # x

            # center of box
            plt.scatter(x=cx, y=cy, c='r')

        if gaussian_map is not None:
            plt.figure('map')
            plt.imshow(gaussian_map.numpy())

        plt.show()
        # plt.savefig('./test_result/result.png')

    def load_img_with_box(self, idx):
        image = Image.open(self.images_path[idx]).convert('RGB')
        img_name = os.path.basename(self.images_path[idx])
        boxes = self.bbox_dict[img_name]
        img_name = img_name.split('.')[0]
        image = FT.to_tensor(image)
        boxes = torch.FloatTensor(boxes)

        return image, boxes

    def __getitem__(self, idx):
        img_name_original = os.path.basename(self.images_path[idx])
        # print(img_name_original)
        img_name = img_name_original.split('.')[0]
        img_name_to_ascii = [ord(c) for c in img_name]
        img_name = torch.FloatTensor([img_name_to_ascii])

        # Load Image
        image = Image.open(self.images_path[idx]).convert('RGB')
        img_width, img_height = float(image.size[0]), float(image.size[1])

        boxes = self.bbox_dict[img_name_original]

        image = FT.to_tensor(image)
        boxes = torch.FloatTensor(boxes)
        image, boxes, dims = resize(image, boxes, dims_800=False)

        height, width = dims
        pyramid_levels = np.array([3, 4, 5, 6, 7])
        feature_maps_y = [(height + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        feature_maps_x = [(width + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        yx = (feature_maps_y[0], feature_maps_x[0])

        image = FT.to_tensor(image)
        additional_info = torch.FloatTensor([img_width, img_height])

        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
        cycx = torch.cat([cxcy[:, 1:2], cxcy[:, 0:1]], dim=1)

        locations = cycx
        counts = torch.tensor(locations.size(0), dtype=torch.get_default_dtype())
        if counts == 0:
            locations = torch.tensor([-1, -1], dtype=torch.get_default_dtype())

        gaussian_map = self.create_gaussian_map(yx, boxes)
        labels = torch.zeros(boxes.size(0), dtype=torch.int64)
        w = image.size(2)
        h = image.size(1)
        wh = torch.FloatTensor([w, h, w, h]).unsqueeze(0)
        boxes = boxes / wh

        image = FT.normalize(image, mean=self.mean, std=self.std)

        if self.visualize:
            self.visualize_box(image, boxes, wh, denormalize=True, gaussian_map=gaussian_map)

        if self.split == "test" or self.split == "val":
            return image, boxes, labels, locations, counts, gaussian_map, img_name, additional_info

        return image, boxes, labels, locations, counts, gaussian_map

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        locations = list()
        counts = list()
        gaussian_map = list()
        img_name = list()
        additional_info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            locations.append(b[3])
            counts.append(b[4])
            gaussian_map.append(b[5])
            if self.split == "test" or self.split == "val":
                img_name.append(b[6])
                additional_info.append(b[7])

        images = torch.stack(images, dim=0)
        gaussian_map = torch.stack(gaussian_map, dim=0)
        counts = torch.stack(counts, dim=0)

        if self.split == "test" or self.split == "val":
            return images, boxes, labels, locations, counts, gaussian_map, img_name, additional_info
        return images, boxes, labels, locations, counts, gaussian_map

    def create_gaussian_map(self, resize, boxes):
        y = resize[0]
        x = resize[1]
        x = np.linspace(0, x, x)
        y = np.linspace(0, y, y)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        for i, box in enumerate(boxes):
            box = box / 8
            cx, cy = (box[:2] + box[2:]) / 2
            w, h = box[2:] - box[:2]
            sigma_w = w
            sigma_h = h
            rv = multivariate_normal(mean=[cx, cy], cov=[[sigma_w, 0], [0, sigma_h]])
            if i == 0:
                pd = rv.pdf(pos).astype(np.float32)
                pd_tensor = torch.from_numpy(pd)
                pd_tensor = (pd_tensor - pd_tensor.min()) / (pd_tensor.max() - pd_tensor.min())
            else:
                pd_new = rv.pdf(pos).astype(np.float32)
                pd_new_tensor = torch.from_numpy(pd_new)
                pd_new_tensor = (pd_new_tensor - pd_new_tensor.min()) / (pd_new_tensor.max() - pd_new_tensor.min())
                pd_tensor = torch.where(pd_tensor < pd_new_tensor, pd_new_tensor, pd_tensor)

        pd_tensor = (pd_tensor - pd_tensor.min()) / (pd_tensor.max() - pd_tensor.min())
        return pd_tensor


if __name__ == '__main__':
    root = 'D:\SKU110K_fixed'
    dataset = SKU110K_Dataset(root=root, split='train', resize=800, visualize=True)
    data = dataset.__getitem__(idx=0)
