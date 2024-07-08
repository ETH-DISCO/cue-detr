import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import torch
import os


def get_slice_borders(w, cue, b):
    if b < 0: # uniformly random
        l = cue - np.random.randint(w - 1)
    elif b == 0: # centered
        l = cue - (w // 2)
    else: # segmentation
        l = cue - (w // b) - np.random.randint(w * (b - 2) // b)
    
    return l, l + w


def get_image_slice(image, l, r):
    """ Returns the image slice with possible borders overlapping the original image """
    if l < 0:
        img_slice = image[:, :r]
        assert img_slice.shape[1] == r, f'Invalid slice width: {img_slice.shape[1]}, should be {r} (l: {l}, r: {r})'
        
        pad = -l
        img_slice = np.pad(img_slice, ((0, 0), (pad, 0), (0, 0)), mode='linear_ramp')
    elif r > image.shape[1]:
        img_slice = image[:, l:]
        assert img_slice.shape[1] == image.shape[1] - l, f'Invalid slice width: {img_slice.shape[1]}, should be {image.shape[1] - l} (l: {l}, r: {r})'
        
        pad = r - l - img_slice.shape[1]
        img_slice = np.pad(img_slice, ((0, 0), (0, pad), (0, 0)), mode='linear_ramp')
    else:
        img_slice = image[:, l:r]
    
    assert img_slice.shape[1] == r - l, f'Invalid slice width: {img_slice.shape[1]}, should be {r - l} (l: {l}, r: {r})'
    return img_slice


def cue_to_bbox(cue_dict, w_box, h_box, l, r):
    """ Adjust bounding box parameters if it overlaps with the image slice borders. """
    
    h = h_box
    w = w_box
    w_sl = r - l
    x = cue_dict['position'] - (w // 2)
    y = 0
    
    if x + w > r:   # bbox overlaps right image border
        w = r - x
    if x < l:       # bbox overlaps left image border
        w = w - (l - x)
        x = l
    x = x - l       # absolute pixel to slice pixels

    assert 0 <= x <= w_sl, f'Invalid bbox: x = {x}, should be within slice of width {w_sl}'
    assert 0 < w <= w_box, f'Invalid bbox: width = {w}, should be within ]0, {w_box}]'
    
    cue_dict['bbox'] = [x, y, w, h]
    cue_dict['area'] = w * h
    cue_dict.pop('position', None)

    return cue_dict


def cxcywh_to_xyxy(center_box):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L532
    # version with torch format
    center_x, center_y, width, height = center_box.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


def draw_image_with_boxes(image, boxes, box_color, is_xyxy=False):
    """ image: PIL image, labels: list of target dicts that contain the 'boxes' key """

    W, H = image.size
    draw = ImageDraw.Draw(image)
    
    # Convert to pixel values
    if not is_xyxy:
        boxes = cxcywh_to_xyxy(boxes)
    boxes *= torch.tensor([W, H, W, H]).type_as(boxes)
    # boxes *= torch.tensor([W, H, W, H]).to(boxes.device)
    
    for box in boxes:
        # skip empty boxes from padding
        if torch.count_nonzero(box) > 0:
            draw.rectangle(tuple(box.tolist()), outline=box_color, width=1)
    return image


def plot_boxes(images, gt_targets, pd_targets, save_dir):
    for i, img in enumerate(images):
        img = draw_image_with_boxes(img, gt_targets[i]['boxes'], 'white')
        img = draw_image_with_boxes(img, pd_targets[i]['boxes'], 'magenta', is_xyxy=True)
        images[i] = img
        
    r = int(np.sqrt(len(images)))
    c = int(np.ceil(len(images) / r))
    images += [Image.new('RGB', size=images[0].size)] * (r*c - len(images))

    grid = plot_grid(images, r, c)
    grid.save(save_dir, 'PNG')
        

def plot_grid(images, rows, cols):
    assert len(images) == rows*cols, f'Number of images ({len(images)}) does not match grid dimension ({rows} x {cols}).'

    offset = 10
    w, h = images[0].size
    W = cols * w + (cols + 1) * offset
    H = rows * h + (rows + 1) * offset
    grid = Image.new('RGB', size=(W, H))

    for n, img in enumerate(images):
        # grid coordinates
        i = (n%cols)
        j = (n//cols)
        # grid to pixel coordinates
        x = i * (w + offset) + offset
        y = j * (h + offset) + offset
        grid.paste(img, box=(x, y))
    
    return grid


def draw_header(image, title):
    W, H = image.size
    canvas = Image.new('RGB', size=(W, H+20))
    canvas.paste(image, box=(0, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), title, fill='white')
    return canvas


def draw_cues(track_image, gt_cues, pd_cues, other_cues=None):
    canvas = ImageDraw.Draw(track_image)
    
    # Draw the predicted cues
    for c in pd_cues:
        # Draw the vertical line
        canvas.line((c, 0, c, 128), fill='magenta', width=2)
    
    # Draw the ground truth cues
    for c in gt_cues:
        # Spacing between the Dots
        for i in range(0, 128, 10):
            # Dots
            canvas.line((c, i, c, i+5), fill='white', width=2)

    # Draw the other cues if provided
    if other_cues is not None:
        for c in other_cues:
            # Spacing between the Dots
            for i in range(0, 128, 20):
                # Dots
                canvas.line((c, i, c, i+10), fill='yellow', alpha=0.5, width=2)

    return track_image


def get_description(model_name, n_gt, n_pd, first, second):
    # Add description onto image
    if second.isnumeric():
        description = f'{model_name}        found: {n_pd} / expected: {n_gt}       topk radius: {second}'
    else:
        description = f'{model_name}        found: {n_pd} / expected: {n_gt}       threshold: {first}, grouping center: {second}'
    return description


def make_scatter_plot(data, thresholds, radius, title, save_dir):
    n, _ = data.shape
    if n != len(radius):
        data = data.T

    _, ax = plt.subplots()
    idx = np.unravel_index(np.argmax(data, axis=None), data.shape)
    area = (50 * data)**2

    for i in range(len(radius)):
        y = i * np.ones(len(thresholds))
        plt.scatter(thresholds, y, s=area[i], c=data[i], alpha=0.5)
        for i, txt in enumerate(data[i]):
            ax.annotate(round(txt, 2), (thresholds[i], y[i]), fontsize=8)
            
    plt.scatter(thresholds[idx[1]], idx[0], s=area[idx], edgecolors='red', facecolors='none')
    plt.xlabel('Threshold')
    plt.ylabel('Radius')
    plt.xticks(np.around(thresholds, 2))
    plt.yticks(np.arange(3), labels=radius)
    plt.ylim(-0.5, 2.5)
    plt.xlim(0, 1)
    plt.title(title)

    plt.savefig(os.path.join(save_dir, f'{title}.png'))


def save_as_csv(thresholds, radius, ap_values, p_values, r_values, file_name):
    header = np.asarray(f'AP, {radius}, P, {radius}, R, {radius}'\
                        .replace('[', '').replace(']', '').split(','))
    dash = np.full(len(thresholds), '-')
    data = np.column_stack((np.around(thresholds, 2), ap_values, dash, p_values, dash, r_values))
    data = np.vstack((header, data))
    np.savetxt(f'{file_name}.csv', data, delimiter=',', fmt='%s')
