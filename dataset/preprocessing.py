#!/usr/bin/python
import argparse

from datetime import datetime
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
import librosa
import json

class Marker(IntEnum):
    CUE = 0
    BEAT = 1
    DBEAT = 2

# COCO Data
#BB_W = 3  # bounding box width = BB_W + 1 + BB_W = 7
#BB_H = MELS
#BB_A = BB_H * (2 * BB_W + 1)


def generate_images(tracks, mel_fft, mel_hop, audio_dir, image_dir):    
    
    images = []
    annotations = []
    annotation_id = 0

    for img_id, track in enumerate(tracks):
        track_id = str(track['id'])

        # MFCC and save as RBG image
        y, sr = librosa.load(os.path.join(audio_dir, track_id + '.mp3'))  # standard sr of 22050
        M = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=mel_fft)
        M_db = librosa.power_to_db(M, ref=np.max)
        plt.imsave(os.path.join(image_dir, track_id + '.png'), M_db, origin='lower')
        img = {
            'id': img_id,
            'width': int(M_db.shape[1]),
            'height': int(M_db.shape[0]),
            'file_name' : track_id + '.png',
        }
        images.append(img)

        # ANNOTATIONS
        cue_frames = sorted(librosa.time_to_frames(track['cue_pts'], sr=22050, n_fft=mel_fft, hop_length=mel_hop))

        for c in cue_frames:
            annotations.append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': Marker.CUE,
                'position': int(c)
                #'bbox': [int(c - BB_W), 0, BB_W * 2 + 1, BB_H],
                #'area': BB_A
            })
            annotation_id+=1

        '''
        # Beat Grid
        t_beat = 60 / track['beat_grid']['bpm']
        duration = track['duration']
        n_beats = int(duration / t_beat)
        beat_grid = [(t_beat * i) for i in range(n_beats)]

        first_beat_nr = track['beat_grid']['init_beat']
        down_beat_grid = beat_grid[first_beat_nr-1::4]

        frame_beats = librosa.time_to_frames(beat_grid, sr=sr, n_fft=n_fft, hop_length=hop_len)
        frame_dbeats = librosa.time_to_frames(down_beat_grid, sr=sr, n_fft=n_fft, hop_length=hop_len)

        for b in frame_beats:
            ann_id+=1
            a.append(annotate(ann_id, i, Marker.BEAT, b, bb_w, bb_h))

        for db in frame_dbeats:
            ann_id+=1
            a.append(annotate(ann_id, i, Marker.DBEAT, db, bb_w, bb_h))
        '''
    return images, annotations



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--annotation_file',
                    help='Out: Name of annotation file (default: annotations.json).',
                    type=str, default='annotations.json')
    ap.add_argument('-d', '--dataset_file',
                    help='Name of dataset json-file for conversion.',
                    type=str)
    ap.add_argument('-a', '--audio_dir',
                    help='Path to directory with audio.',
                    type=str, required=True)
    ap.add_argument('-i', '--image_dir',
                    help='Path to directory where images will be saved.',
                    type=str, required=True)
    ap.add_argument('-o', '--prev_images',
                    help='Previous annotation file for updating coco file.',
                    type=str)
    ap.add_argument('-f', '--fft',
                    help='n_fft (default = 2048).',
                    type=int, default=2048)
    ap.add_argument('-c', '--contributor',
                    help='Name of the contributor.',
                    type=str, default='None')

    args = ap.parse_args()
    file_name = args.annotation_file
    track_list = args.dataset
    dir_audio = args.audio_dir
    dir_images = args.image_dir
    old_images = args.old_images

    mel_fft = args.fft
    mel_hop = mel_fft // 4

    with open(track_list, 'r') as f:
        tracks = json.load(f)['tracks']

    images, annotations = generate_images(tracks, mel_fft, mel_hop, dir_audio, dir_images)

    data = {
        'info' : {
            'description': 'MS COCO-like Dataset with Cue Point Annotations',
            'url': '',
            'version': '0.2',
            'year': datetime.today().year,
            'contributor': args.contributor,
            'date_created': datetime.today().strftime('%Y-%m-%d')
        },
        'images' : images,
        'annotations': annotations,
        'categories': [{
            'id': Marker.CUE,
            'name': 'cue',
            'supercategory' : 'cue'
        }]
    }

    with open(file_name, 'w') as f:
        json.dump(data, f)