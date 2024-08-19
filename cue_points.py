from transformers import DetrImageProcessor, DetrForObjectDetection
from scipy.signal import find_peaks
import torch

from matplotlib import cm
from PIL import Image
import numpy as np
import argparse
import librosa
import os

# CONSTANTS
OVERLAP = 0.75
W_BBOX = 21
W_WIN = 355
PADDING = 266

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict cue points for one or more tracks using CUE-DETR.\
                The resulting _cue_points.txt will be saved in the directory with all tracks.")
    parser.add_argument('-t', '--tracks', type=str, required=True, help="Path to track directory")
    parser.add_argument('-c', '--checkpoint', type=str, default='disco-eth/cue-detr', help="Optional local path to model checkpoint")
    parser.add_argument('-s', '--sensitivity', type=float, default=0.9, help="Threshold value for cue points (default = 0.9)")
    parser.add_argument('-r', '--radius', type=int, default=16, help="Minimum distance in bars between cue points in bars (default = 16)")
    parser.add_argument('-p', '--print', action='store_true', help="Print cue points to console")
    args = parser.parse_args()

    tracklist = [file for file in os.listdir(args.tracks) if file.endswith('.mp3')]
    cue_points = {track: [] for track in tracklist}
    scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    # Load Model
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DetrForObjectDetection.from_pretrained(args.checkpoint)
    model.to(device)

    for track in tracklist:
        y, sr = librosa.load(os.path.join(args.tracks, track))  # standard sr of 22050
        M = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048)
        M_db = librosa.power_to_db(M, ref=np.max)

        # Convert to RGB image without saving (plt.saveimage)
        arr = M_db[::-1]
        sm = cm.ScalarMappable(cmap='viridis')
        sm.set_clim(arr.min(), arr.max())
        rgba = sm.to_rgba(arr, bytes=True)
        rgb_shape = (rgba.shape[1], rgba.shape[0])
        rgba = np.require(rgba, requirements='C')
        im = Image.frombuffer("RGBA", rgb_shape, rgba, "raw", "RGBA", 0, 1)
        image = np.array(im)
        image = image[:, :, :3]

        image_w = image.shape[1]
        image_w += PADDING
        n_windows = int(np.floor(image_w / (W_WIN * (1 - OVERLAP))))
        
        images = []
        borders = []

        # Create image batch with sliding window
        for i in range(n_windows):
            l = int(np.floor(i * W_WIN * (1 - OVERLAP))) - PADDING
            r = l + W_WIN
            borders.append(l)

            # Compute image segment
            if l < 0:
                segment = image[:, :r]        
                pad = -l
                segment = np.pad(segment, ((0, 0), (pad, 0), (0, 0)), mode='linear_ramp')
            elif r > image.shape[1]:
                segment = image[:, l:]        
                pad = r - l - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad), (0, 0)), mode='linear_ramp')
            else:
                segment = image[:, l:r]
            
            images.append(segment)
    
        # Preprocess images
        encoding = image_processor.preprocess(images, do_resize=False, return_tensors='pt')
        pixel_values = encoding['pixel_values']
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            outputs = model(pixel_values)
        
        # Convert to scores, labels, boxes (in pixel coordinates)
        to_pixel = [(128, 355)] * pixel_values.shape[0]
        predictions = image_processor.post_process_object_detection(outputs, 0, to_pixel)

        # Convert to box centers
        scores = []
        positions = []
        for p, l in zip(predictions, borders):
            scores.extend(p['scores'].tolist())
            # box -> cue -> spectrogram
            pos = (p['boxes'][:, 0] + p['boxes'][:, 2]) // 2 + l
            positions.extend(pos.long().tolist())

        # Order by position
        positions, scores = zip(*sorted(zip(positions, scale(scores))))

        # Find peaks and add to cue point collection
        peak_idx, _ = find_peaks(scores, height=args.sensitivity, distance=args.radius)
        cue_positions = [positions[idx] for idx in peak_idx]
        cue_points[track] = list(librosa.frames_to_time(cue_positions))
        if args.print:
            print(f"{track}: {cue_points[track]}")

    # Save cue points to file
    with open(os.path.join(args.tracks, '_cue_points.txt'), 'w') as f:
        for track in cue_points:
            f.write(f"{track}: {cue_points[track]}\n")