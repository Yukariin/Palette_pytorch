import os

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


class DS(data.Dataset):
    def __init__(self, root, transform=None, irregular_mask_kwargs={}, box_mask_kwargs={}):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.transform = transform
        self.irregular_mask_kwargs = irregular_mask_kwargs
        self.box_mask_kwargs = box_mask_kwargs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = Image.open(sample_path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if np.random.random() > 0.5:
            mask = self.irregular_mask(**self.irregular_mask_kwargs)
        else:
            mask = self.rectangle_mask(**self.box_mask_kwargs)
        mask = torch.from_numpy(mask)

        return sample, mask

    @staticmethod
    def irregular_mask(height=256, width=256,
                    min_times=0, max_times=10,
                    max_vertex=5,
                    max_angle=4,
                    max_len=60,
                    max_width=20):
        mask = np.zeros((height, width), np.float32)

        times = np.random.randint(min_times, max_times+1)
        for i in range(times):
            start_x = np.random.randint(height)
            start_y = np.random.randint(width)

            num_vertex = 1 + np.random.randint(max_vertex)
            for j in range(num_vertex):
                angle = 0.01 + np.random.randint(max_angle)
                if j % 2 == 0:
                    angle = 2*np.pi - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, height)
                end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, width)

                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1., brush_w)

                start_x, start_y = end_x, end_y

        return mask[None, ...]

    @staticmethod
    def rectangle_mask(height=256, width=256,
                       margin=10,
                       min_times=0, max_times=3,
                       min_size=30, max_size=100):
        mask = np.zeros((height, width), np.float32)

        max_size = min(max_size, height - margin * 2, width - margin * 2)
        times = np.random.randint(min_times, max_times+1)
        for i in range(times):
            box_width = np.random.randint(min_size, max_size)
            box_height = np.random.randint(min_size, max_size)
            start_x = np.random.randint(margin, width - margin - box_width + 1)
            start_y = np.random.randint(margin, height - margin - box_height + 1)
            mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1.

        return mask[None, ...]

