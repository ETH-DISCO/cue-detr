from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import glob, os

# project modules
from cue_detr_model import CuePointDetr
from cue_detr_data import CuePredictDataset

class CuePointPredictor():
    def __init__(self, image_dir, checkpoint, image_processor=None, **kwargs):
        
        # Image Processor (Tokenizer)
        self.image_processor = None
        if image_processor is not None:
            self.image_processor = DetrImageProcessor.from_pretrained(image_processor)
        else:
            self.image_processor = DetrImageProcessor(size={'longest_edge': 355, 'shortest_edge': 128}, do_resize=False, do_pad=False)
        assert self.image_processor is not None, 'Image processor initialization failed.'
        
        # Dataset
        image_names = kwargs['images']\
            if 'images' in kwargs and kwargs['images'] is not None\
            else [os.path.split(f)[1] for f in glob.glob(image_dir + '*.png')]
        
        self.dataset = CuePredictDataset(image_names, image_dir, self.image_processor)

        if 'overlap' in kwargs and kwargs['overlap'] is not None:
            self.dataset.set_window_overlap(kwargs['overlap'])
        if 'w_window' in kwargs and kwargs['w_window'] is not None:
            self.dataset.set_window_width(kwargs['w_window'])
        if 'w_bbox' in kwargs and kwargs['w_bbox'] is not None:
            self.dataset.set_box_width(kwargs['w_bbox'])

        # Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if checkpoint.endswith('.ckpt'):
            self.model = CuePointDetr.load_from_checkpoint(checkpoint)
        else:
            self.model = DetrForObjectDetection.from_pretrained(checkpoint)
        self.model.to(self.device)

    
    def predict_over(self, track_idx):
        """
        Returns prediction in a dict with keys: 'scores', 'positions'
        """
        pixel_values = self.dataset[track_idx]
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)

        # Convert to scores, labels, boxes (in pixel coordinates)
        to_pixel = [(128, 355)] * pixel_values.shape[0]
        predictions = self.image_processor.post_process_object_detection(outputs, 0, to_pixel)

        # Convert to box centers
        track_id = self.dataset.get_image_id(track_idx)
        left_pad = self.dataset.get_left_offset(track_id)
        
        borders = [self.dataset.get_window_border(i, left_pad)[0] for i in range(len(predictions))]
        scores = []
        positions = []
        for p, l in zip(predictions, borders):
            scores.extend(p['scores'].tolist())
            # box -> cue -> spectrogram
            pos = (p['boxes'][:, 0] + p['boxes'][:, 2]) // 2 + l
            positions.extend(pos.long().tolist())
        
        return {
            'scores': scores,
            'positions': positions,
        }


    def predict_all(self):
        """
        Returns prediction in a dict mapping file name to dict with keys: 'scores', 'positions'
        """
        all_predictions = {}
        for track_idx in range(len(self.dataset)):
            # Save results mapped to the image file name (without extension)
            image_id = self.dataset.get_image_id(track_idx)
            file = self.dataset.get_image_name(image_id)
            # all_predictions[file] = self.predict_over(track_idx)
            all_predictions[int(file[:-4])] = self.predict_over(track_idx)

        return all_predictions

