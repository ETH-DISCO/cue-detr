# CUE-DETR (repo work in progress)

[📜Paper](https://www.arxiv.org/abs/2407.06823) | [🤗Dataset](https://huggingface.co/datasets/disco-eth/edm-cue) | [🤗Checkpoints](https://huggingface.co/disco-eth/cue-detr/tree/main)

## Dataset

CUE-DETR accepts data with annotations in a modified COCO format. The dataset contains a `'position'` parameter instead of `'bbox'` and `'area'` for annotations since bounding boxes will be computed during runtime. The script `spectrograms.py` converts our data into the required format.

<details>
<summary> Custom COCO Format </summary>

```python
data = {
    'images' : [{
        'id': img_id,
        'width': int,
        'height': int,
        'file_name' : filename,
    }]
    'annotations': [{
        'id': annotation_id,
        'image_id': img_id,
        'category_id': 0,
        'position': int # cue position instead of bounding box
    }],
    'categories': [{
        'id': 0,
        'name': 'cue',
        'supercategory' : 'cue'
    }]
}
```
</details>


## Training

Uses W&B for logging. Connect to W&B account by running `wandb login` in the console and passing the projectname and account as arguments for training.



## Dependencies

Python 3.11.9, see `requirements.txt`.
