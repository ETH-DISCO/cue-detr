from transformers import DetrForObjectDetection

# pytorch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torch import nn
import torch

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
import wandb

from PIL import Image

from cue_detr_utils import draw_image_with_boxes

class CuePointDetr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, **kwargs):
        super().__init__()
        if 'config' in kwargs and kwargs['config'] is not None:
            self.model = DetrForObjectDetection(kwargs['config'])
        elif 'detr_checkpoint' in kwargs and kwargs['detr_checkpoint'] is not None:
            self.model = DetrForObjectDetection.from_pretrained(
                kwargs['detr_checkpoint'],
                num_labels=1,
                ignore_mismatched_sizes=True,
                )
        assert self.model is not None, 'Model initialization failed.'
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss, loss_dict = outputs.loss, outputs.loss_dict  
        
        self.log("training_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"training_{k}", v, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss, loss_dict = outputs.loss, outputs.loss_dict
        
        self.log("validation_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v, on_step=False, on_epoch=True, sync_dist=True)     
        return loss
    
    def configure_optimizers(self):
        """ defines model optimiser """
        param_dicts = [
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
                },
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                }
        ]
        
        optimizer = AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "validation_loss",
            }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    

class CuePredictionLogger(Callback):
    def __init__(self, sample_idx, dataset):
        super().__init__()
        pixel_values, target = dataset[sample_idx]
        l, r = dataset.last_slice
        image = dataset.images[target['image_id'].item()]['file_name']
        image = Image.open(image).convert('RGB').crop((l, 0, r, 128))

        self.pixel_values = torch.stack([pixel_values])
        self.target = target
        self.image = image


    def on_validation_epoch_end(self, trainer, pl_module):       
        # only log on rank 0 when using distributed training
        if rank_zero_only.rank != 0:
            return
        
        # push data onto device
        device = pl_module.device
        pixel_values = self.pixel_values.to(device)
        labels = [{k: v.to(device) for k, v in self.target.items()}]
        
        # pass once through the model to get predictions
        with torch.no_grad():
            outputs = pl_module(pixel_values=pixel_values, labels=labels)

        prob = nn.functional.softmax(outputs.logits, -1)
        scores, _ = prob[..., :-1].max(-1)
        
        scores, idx = scores.topk(5)
        idx = torch.flatten(idx).tolist()
        pd_boxes = outputs.pred_boxes[:, idx, :].squeeze(0)
        gt_boxes = torch.stack([l['boxes'] for l in labels]).flatten(1)

        # draw on image
        image = self.image.copy()
        image = draw_image_with_boxes(image, pd_boxes, box_color='red')
        image = draw_image_with_boxes(image, gt_boxes, box_color='white')
        trainer.logger.experiment.log({'example_image': wandb.Image(image)}, commit=False)



'''
================================
        HELPER FUNCTIONS
================================
'''
def to_wandb_image_annotation(box, class_id, scores):
    x, y, w, h = tuple(box)
    return { 
        'position': {
            'middle': [x, y],
            'width': w,
            'height': h,
            },
        'class_id' : class_id,
        'scores' : scores
    }


