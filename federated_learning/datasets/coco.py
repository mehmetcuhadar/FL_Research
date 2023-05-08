import torch
from torchvision import transforms
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

class CocoDataset:

    def __init__(self, args):
        self.args = args
        self.catIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # IDs of the 10 object categories

    def load_train_dataset(self):
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        train_dataset = CocoDetection(root=self.args.data_path, annFile=self.args.train_ann_file,
                                       transform=transform, target_transform=None, catIds=self.catIds)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        train_data = next(iter(train_loader))

        return train_data

    def load_test_dataset(self):
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        test_dataset = CocoDetection(root=self.args.data_path, annFile=self.args.val_ann_file,
                                      transform=transform, target_transform=None, catIds=self.catIds)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        test_data = next(iter(test_loader))

        return test_data
