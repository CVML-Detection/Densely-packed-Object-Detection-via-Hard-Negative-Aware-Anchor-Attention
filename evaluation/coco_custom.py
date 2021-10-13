__author__ = 'skrish'
__version__ = '0.1'


class CustomCOCO:
    def __init__(self, list_of_coco_dicts):
        self.list_of_coco_dicts = list_of_coco_dicts

    def getImgIds(self):
        return list(set([e['image_id'] for e in self.list_of_coco_dicts]))

    def getCatIds(self):
        return list(set([e['category_id'] for e in self.list_of_coco_dicts]))

    def getAnnIds(self, imgIds, catIds):
        AnnIds = [element['id'] for element in self.list_of_coco_dicts if
                  element['image_id'] in imgIds and element['category_id'] in catIds]
        return AnnIds

    def loadAnns(self, AnnIds):
        list_of_filtered_dicts = [element for element in self.list_of_coco_dicts if element['id'] in AnnIds]
        return list_of_filtered_dicts

