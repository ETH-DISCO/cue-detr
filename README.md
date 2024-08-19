# CUE-DETR

[📜Paper](https://www.arxiv.org/abs/2407.06823) | [🤗Dataset](https://huggingface.co/datasets/disco-eth/edm-cue) | [🤗Checkpoints](https://huggingface.co/disco-eth/cue-detr/tree/main)

## Dataset

[🤗Dataset](https://huggingface.co/datasets/disco-eth/edm-cue)

***EDM-CUE*** contains metadata for almost 5k EDM tracks collected from 4 different DJs. No audio provided, only references to training data.

<details>
<summary> Track Metadata </summary>

```python
{
    'id': int,
    'title': str,
    'artists': str,
    'duration': int,        # in seconds
    'genre': [str],
    'key': [str],           # alphanumeric (Camelot)
    'beat_grid': {
        'start_pos': float, # in seconds
        'init_beat': int,   #  first beat count
        'bpm': float,
        'time_sig': str
    },
    'cue_pts': [float]      # in seconds
}
```
</details>

#### Training Format

* CUE-DETR expects training data in a ***modified COCO format***: instead of `'bbox'` and `'area'` the model requires the `'position'` of each cue point annotation. The bounding box is computed during runtime with default width 21 pixels.

* `preprocessing.py` converts audio into power spectrograms including the annotation file in the custom COCO format.

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

See `cue_detr_train.py`, `cue_detr_data.py` and `cue_detr_model.py` in `model` directory.


## Dependencies

Python 3.11.9, see `requirements.txt`.


## Usage / Example Script

[🤗Checkpoints](https://huggingface.co/disco-eth/cue-detr/tree/main)

The example script `cue_points.py` calculates cue points for tracks stored in an audio directory. All calculated cue points will be written to `_cue_points.txt` which is added to the audio directory. It is also possible to run the script with a local checkpoint from a checkpoint directory. Note that as of now only mp3 files are supported.

```bash
python cue_points.py -t path/to/audio/dir
# Optional arguments:
# -c (path/to/local/checkpoint/dir)
# -s (prediction sensitivity)
# -r (min distance between cues)
# -p (toggle to print cue points)
```

