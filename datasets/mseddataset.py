import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transform import get_transform
from .utils import read_json


class MSEDDataset(Dataset):
    LABEL_MAPPING = {
        'sentiment': {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        },
        'emotion': {
            'happiness': 0,
            'sad': 1,
            'neutral': 2,
            'disgust': 3,
            'anger': 4,
            'fear': 5
        },
        'desire': {
            'vengeance': 0,
            'curiosity': 1,
            'social-contact': 2,
            'family': 3,
            'tranquility': 4,
            'romance': 5,
            'none': 6
        }
    }

    def __init__(self, root_path, img_process, tokenizer, train_type='train', label_type='sentiment'):
        Dataset.__init__(self)
        self.root_path = root_path
        self.img_process = img_process
        self.tokenizer = tokenizer
        self.label_type = label_type
        self.train_type = train_type
        self.data_list = self._read_file(root_path)
        self.transform_448 = get_transform("padded_resize", 448)

    def _read_file(self, path):
        return read_json(f"{path}/{self.train_type}.json")

    def _split_into_patches(self, img_448: torch.Tensor) -> torch.Tensor:
        """
        448x448 images split into 4x224x224 sub-images
        """
        # patches = img_448.unfold(1, 224, 224).unfold(2, 224, 224)
        #
        # # [C, num_patches_h, num_patches_w, patch_h, patch_w]
        # # -> [num_patches, C, patch_h, patch_w]
        # patches = patches.reshape(3, 2, 2, 224, 224)
        # patches = patches.permute(1, 2, 0, 3, 4)  # [2, 2, 3, 224, 224]
        # patches = patches.reshape(4, 3, 224, 224)  # [4, 3, 224, 224]
        #
        # return patches
        return torch.stack([
            img_448[:, :224, :224],
            img_448[:, :224, 224:],
            img_448[:, 224:, :224],
            img_448[:, 224:, 224:]
        ], dim=0)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        text, img, label = data['caption'], data['image'], data[self.label_type]

        img_path = os.path.join(self.root_path, self.train_type, 'images', img)
        image = Image.open(img_path).convert('RGB')

        img_224, img_448 = self.img_process(image)['pixel_values'][0], self.transform_448(image)
        img_448_patches = self._split_into_patches(img_448)
        text_token = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77  # self.config.model.text_decoder['context_length']
        )

        return {
            'img_path': int(img.split('.')[0]),
            'img_224': img_224,
            'raw_img_448': img_448,
            'img_448': img_448_patches,
            'input_ids': text_token['input_ids'],
            'attention_mask': text_token['attention_mask'],
            'labels': self.LABEL_MAPPING[self.label_type][label],
        }

    def __len__(self):
        return len(self.data_list)

    def _get_class_name(self):
        return list(self.LABEL_MAPPING[self.label_type].keys())

