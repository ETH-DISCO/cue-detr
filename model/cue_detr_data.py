# dataset
from torch.utils.data import Dataset
from transformers import DetrImageProcessor

# lightning
import pytorch_lightning as pl

# dataloader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from copy import deepcopy
from PIL import Image
import numpy as np
import json, glob, os

from cue_detr_utils import get_slice_borders, get_image_slice, cue_to_bbox

class CueTrainDataset(Dataset):
    def __init__(self, img_names, img_folder, image_processor):
        """
        Dataset where a sample is an image slice around a cue point. Expects a dataset split or full dataset.
        For splits, the data should be divided based on the tracks and not annotations.

        Args:
            img_names (list):
                list of image names of the current dataset (split)
            img_folder (str):
                path to the folder containing the images and `annotations.json`
            image_processor (DetrImageProcessor):
                preprocesses images (normalize, pad, resieze, ...) and annotations
        """
        assert os.path.exists(f'{img_folder}/annotations.json'), f'No `annotations.json` file found in {img_folder}'
        with open(f'{img_folder}/annotations.json', 'r') as f:
            coco = json.load(f)
        
        # image ids from current dataset (split)
        self.image_ids = sorted([i['id'] for i in coco['images'] if i['file_name'] in img_names])
        self.images = {i: [] for i in self.image_ids}           # image_id -> {image}
        self.image_to_ann = {i: [] for i in self.image_ids}     # image_id -> [image's ann_ids]
        
        # annotation and image ids from current dataset (split)
        self.ann_ids = []
        self.annotations = {}           # ann_id -> {annotation}

        # images can be accessed with the image id
        for img in coco['images']:
            if img['id'] in self.image_ids:
                # update file name if not full location yet
                if img_folder not in img['file_name']:
                    img['file_name'] = img_folder + img['file_name']
                self.images[img['id']] = img

        # image annotaions can be accessed with the annotation id
        for ann in coco['annotations']:
            if ann['image_id'] in self.image_ids:
                if ann['position'] < 0:
                    ann['position'] = 0
                self.ann_ids.append(ann['id'])
                self.annotations[ann['id']] = ann
                self.image_to_ann[ann['image_id']].append(ann['id'])

        self.categories = [i for i in coco['categories']]
        self.image_processor = image_processor
        
        # Extra params
        self.reuse_slice = False
        self.w_bbox = 7
        self.w_slice = 355
        self.box_area = 4


    def __len__(self) -> int:
        return len(self.ann_ids)


    def __getitem__(self, idx):

        ann_id = self.ann_ids[idx]
        curr_ann = deepcopy(self.annotations[ann_id])
        image_id = curr_ann['image_id']
        image = np.asarray(Image.open(self.images[image_id]['file_name']).convert('RGB'))
        
        boxes = []
        if self.reuse_slice and 'slice' in curr_ann.keys():
            l, r = curr_ann['slice']['lr']
            boxes = curr_ann['slice']['boxes']
            curr_ann.pop('slice')
        else:
            # fix cue to start of image
            c = curr_ann['position']
            l, r = get_slice_borders(self.w_slice, c, b=self.box_area)
            
            # see if the slice contains more annotations
            track_ann_ids = self.image_to_ann[image_id]
            h = self.images[image_id]['height']
            for i in track_ann_ids:
                if l <= self.annotations[i]['position'] <= r:
                    cue = deepcopy(self.annotations[i])
                    boxes.append(cue_to_bbox(cue, self.w_bbox, h, l, r))

            # keep for next time in original annotation
            if self.reuse_slice:
                self.annotations[ann_id]['slice'] = {
                    'lr': (l, r), 
                    'boxes': boxes}

        assert len(boxes) > 0, f'No cue points in slice [{l}, {r}] based on cue point at {c}.'

        img_slice = get_image_slice(image, l, r)

        assert r - l == self.w_slice, f'Invalid slice width: {r - l}, should be {self.w_slice} (l: {l}, r: {r})'
        assert img_slice.shape[1] == self.w_slice, f'Invalid slice width: {img_slice.shape[1]}, should be {r - l} (l: {l}, r: {r})'

        self.last_slice = (l, r)
        self.last_boxes = boxes

        # target = {image_id, list of all annotations in the image}
        targets = {'image_id': image_id, 'annotations': boxes}
        
        encoding = self.image_processor.preprocess(img_slice, targets, do_resize=False, return_tensors='pt') # pt => TensorType.PYTORCH
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    
    def set_reuse_slice(self, val):
        self.reuse_slice = val

    def set_box_area(self, val):
        self.box_area = val

    def set_box_width(self, w):
        self.w_bbox = w
    
    def set_window_width(self, w):
        self.w_slice = w

    def get_id2label(self):
        return {cat['id']: cat['name'] for cat in self.categories}
    
    def get_label2id(self):
        return {cat['name']: cat['id'] for cat in self.categories}
    
    def get_image_id(self, idx):
        return self.annotations[self.ann_ids[idx]]['image_id']


class CuePredictDataset(Dataset):
    def __init__(self, img_names, img_folder, image_processor):
        """
        Dataset used for cue point prediction. Spectrogram images are divided into smaller images using sliding window.

        Args:
            img_names (list):
                list of image names of the current dataset (split)
            img_folder (str):
                path to the folder containing the images and `annotations.json`
            image_processor (DetrImageProcessor):
                preprocesses images (normalize, pad, resieze, ...) and annotations
        """
        assert os.path.exists(f'{img_folder}/annotations.json'), f'No `annotations.json` file found in {img_folder}'
        with open(f'{img_folder}/annotations.json', 'r') as f:
            coco = json.load(f)
        
        # image ids from current dataset (split)
        self.image_ids = sorted([i['id'] for i in coco['images'] if i['file_name'] in img_names])
        self.images = {i: {} for i in self.image_ids}           # image_id -> {image}
        self.annotations = {i: [] for i in self.image_ids}      # image_id -> [{annotation}]
        

        # images can be accessed with the image id
        for img in coco['images']:
            if img['id'] in self.image_ids:
                # update file name if not full location yet
                if img_folder not in img['file_name']:
                    img['file_name'] = img_folder + img['file_name']
                self.images[img['id']] = img
        
        # image annotaions can be accessed with the image id
        for ann in coco['annotations']:
            if ann['image_id'] in self.image_ids:
                if ann['position'] < 0:
                    ann['position'] = 0
                self.annotations[ann['image_id']].append(ann)

        self.image_processor = image_processor
        self.categories = [i for i in coco['categories']]   
        self.window_w = 355
        self.window_h = 128
        self.w_bbox = 7 
        self.overlap = 0.5  


    def __len__(self) -> int:
        return len(self.image_ids)


    def __getitem__(self, idx):
        image_id = self.get_image_id(idx)
        image = self.get_image(image_id)
        image_w, _ = image.size
        image = np.array(image)
        
        # windows
        left_pad = self.get_left_offset(image_id)
        image_w += left_pad
        n_windows = int(np.floor(image_w / (self.window_w * (1 - self.overlap))))

        # targets = []
        images = []
        for i in range(n_windows):
            l, r = self.get_window_border(i, left_pad)
            window = get_image_slice(image, l, r)
            images.append(window)

            # # check if there are annotations in the window
            # labels = []
            # for ann in self.annotations[image_id]:
            #     if l <= ann['position'] <= r:
            #         labels.append(cue_to_bbox(deepcopy(ann), self.w_bbox, image_h, l, r))
            # targets.append({'image_id': image_id, 'annotations': labels})
        encoding = self.image_processor.preprocess(images, do_resize=False, return_tensors='pt')
        
        return encoding['pixel_values']#, encoding['labels'], images
        
    
    def get_image_id(self, idx):
        return self.image_ids[idx]
    

    def get_image(self, image_id):
        return Image.open(self.images[image_id]['file_name']).convert('RGB')
    

    def get_image_name(self, image_id):
        return os.path.split(self.images[image_id]['file_name'])[1]
    

    def get_positions(self, idx):
        image_id = self.get_image_id(idx)
        return [ann['position'] for ann in self.annotations[image_id]]


    def get_window_border(self, window_num, pad):
        l = int(np.floor(window_num * self.window_w * (1 - self.overlap))) - pad
        r = l + self.window_w
        return l, r

    def get_window(self, image_idx, window_num):
        image_id = self.image_ids[image_idx]
        image = self.get_image(image_id)
        image = np.array(image)

        left_pad = self.get_left_offset(image_id)  
        l, r = self.get_window_border(window_num, left_pad)
        window = get_image_slice(image, l, r)
        return window
    

    def get_left_offset(self, image_id):
        # read or generate left offset for whole spectrogram
        if 'left_offset' in self.images[image_id]:
            left_offset = self.images[image_id]['left_offset']
        else:
            left_offset = np.random.randint(self.window_w // 2)
            left_offset += self.window_w // 4
            self.images[image_id]['left_offset'] = left_offset
        
        return left_offset
    
    
    def set_window_overlap(self, val):
        # overlap should be ]0, 1[ 
        eps = 1e-10
        self.overlap = max(eps, min(val, 1-eps))


    def set_window_width(self, w):
        self.window_w = w


    def set_box_width(self, w):
        self.w_bbox = w


class CueTrainModule(pl.LightningDataModule):
    def __init__(self, image_dir, batch_size, **kwargs):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size

        if kwargs['image_processor'] is not None:
            self.image_processor = DetrImageProcessor.from_pretrained(kwargs['image_processor'])
        else:
            self.image_processor = DetrImageProcessor(
                size={'longest_edge': 355, 'shortest_edge': 128},
                do_resize=False,
                do_pad=False)
            
        self.settings = {}

        if 'slice_reuse' in kwargs:
            self.settings['reuse_slice'] = kwargs['slice_reuse']
        if 'w_slice' in kwargs:
            self.settings['w_slice'] = kwargs['w_slice']
        if 'w_bbox' in kwargs:
            self.settings['w_bbox'] = kwargs['w_bbox']
        if 'box_area' in kwargs:
            self.settings['box_area'] = kwargs['box_area']
        if 'pipeline_test' in kwargs:
            self.settings['pipeline_test'] = kwargs['pipeline_test']

    def prepare_data(self):
        # here the dataset can be prepared e.g. slices can be generated and saved to disk
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            image_names = [os.path.split(f)[1] for f in glob.glob(self.image_dir + '*.png')]

            if 'pipeline_test' in self.settings and self.settings['pipeline_test'] is True:
                # use only a subset of the data for testing the pipeline
                train_images = image_names[:self.batch_size]
                val_images = image_names[self.batch_size:2*self.batch_size]
            else:
                # use a split of the data for training and validation
                train_images, val_images = train_test_split(image_names, test_size=0.1, random_state=42)

            self.train_dataset = CueTrainDataset(train_images, self.image_dir, self.image_processor)
            self.val_dataset = CueTrainDataset(val_images, self.image_dir, self.image_processor)

            if 'reuse_slice' in self.settings and self.settings['reuse_slice'] is not None:
                self.train_dataset.set_reuse_slice(self.settings['reuse_slice'])
                self.val_dataset.set_reuse_slice(self.settings['reuse_slice'])
            if 'w_slice' in self.settings and self.settings['w_slice'] is not None:
                self.train_dataset.set_window_width(self.settings['w_slice'])
                self.val_dataset.set_window_width(self.settings['w_slice'])
            if 'w_bbox' in self.settings and self.settings['w_bbox'] is not None:
                self.train_dataset.set_box_width(self.settings['w_bbox'])
                self.val_dataset.set_box_width(self.settings['w_bbox'])
            if 'box_area' in self.settings and self.settings['box_area'] is not None:
                self.train_dataset.set_box_area(self.settings['box_area'])
                self.val_dataset.set_box_area(self.settings['box_area'])
        




    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=1,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=1,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=collate_fn,
        )


def collate_fn(batch):
    return {
        'pixel_values' : torch.stack([item[0] for item in batch]),
        'labels' : [item[1] for item in batch]
    }
