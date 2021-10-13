import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from config import device


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='retina'):
        self.model_name = model_name.lower()
        assert model_name in ['retina']

    @abstractmethod
    def create_anchors(self):
        pass


class RETINA_Anchor(Anchor):
    def create_anchors(self, img_size):
        if type(img_size) is tuple and len(img_size) == 2:
            height = img_size[0]
            width = img_size[1]
        else:
            height = img_size
            width = img_size

        print('make retina anchors')
        pyramid_levels = np.array([3, 4, 5, 6, 7])
        feature_maps_y = [(height + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        feature_maps_x = [(width + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]

        aspect_ratios = np.array([1.0, 2.0, 0.5])
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        strides = [width//f for f in feature_maps_x]
        areas = [32, 64, 128, 256, 512]

        center_anchors = []
        for f_map_x, f_map_y, area, stride in zip(feature_maps_x, feature_maps_y, areas, strides):
            for i in range(f_map_y):
                for j in range(f_map_x):
                    c_x = (j + 0.5) / f_map_x
                    c_y = (i + 0.5) / f_map_y

                    for aspect_ratio in aspect_ratios:
                        for scale in scales:
                            w = (area / width) * np.sqrt(aspect_ratio) * scale
                            h = (area / height) / np.sqrt(aspect_ratio) * scale

                            anchor = [c_x,
                                      c_y,
                                      w,
                                      h]
                            center_anchors.append(anchor)

        center_anchors = np.array(center_anchors).astype(np.float32)
        center_anchors = torch.FloatTensor(center_anchors).to(device)
        return center_anchors


if __name__ == '__main__':

    retina_anchor = RETINA_Anchor(model_name='retina')
    anchor = retina_anchor.create_anchors(img_size=(1333, 800))
    print(len(anchor))

