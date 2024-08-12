# CUE-DETR

PAPER


Model checkpoints



## Dataset

No audio provided, only references.

CUE-DETR accepts data with annotations in a modified COCO format. The dataset contains a `'position'` parameter instead of `'bbox'` and `'area'` for annotations since bounding boxes will be computed during runtime. 

* `preprocessing.py` converts audio into spectrograms including the custom COCO annotation file

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


# Usage

The example script `cue_points.py` calculates cue points for tracks in a directory. Calculated cue points will be written to `_cue_points.txt` which is added to the audio directory. Note that as of now only mp3 files are supported.

The model checkpoint should be stored in a separate directory for which the path must also be passed as script argument.

```bash
python cue_points.py -t path/to/audio/dir -c path/to/checkpoint/dir
# Optional arguments:
# -s (prediction sensitivity)
# -r (min distance between cues)
# -p (toggle to print cue points)
```

