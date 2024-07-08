from scipy.signal import find_peaks
from sklearn.metrics import *

from PIL import Image
import numpy as np
import pandas as pd
import argparse
import json, os
import librosa

# project modules
from cue_detr_predict import CuePointPredictor
from cue_detr_utils import *

'''
========================================
            Main Logcic
========================================
'''
def predict_all(args, model, w=None):
    assert os.path.join(args.result_dir, model), f'Eval folder for model {model} does not exist.'
    
    # Set up model path and add .ckpt if provided
    checkpoint = os.path.join(args.model_dir, model)
    if args.checkpoint is not None:
        checkpoint = os.path.join(checkpoint, args.checkpoint)

    predictor = CuePointPredictor(
        image_dir=args.image_dir,
        checkpoint=checkpoint,
        image_processor=args.image_processor,
        overlap=args.overlap,
        w_window=args.window_width,
        w_bbox=w,
        )
    assert predictor is not None, 'Predictor initialization failed.'
    assert predictor.model is not None, 'Model initialization failed.'
    assert predictor.image_processor is not None, 'Image processor initialization failed.'
    assert len(predictor.dataset) > 0, 'Dataset is empty'

    # predict
    predictions = predictor.predict_all()
    return predictions

    '''
    del predictor

    # save data for later
    print(f'Saving predictions for model {args.model}')
    filepath = os.path.join(args.result_dir, f'predictions.json')
    with open(filepath, 'w') as f:
        json.dump(predictions, f)
    '''


def post_process_score(w_group, thresholds, strategies, fixed_groups, result_dir):
    """
    Post process the predictions of a model and save the results. Filters predictions based on a score threshold
    and groups close by predictions into clusters where the center based on the strategies provided is chosen.

    Args:
        w_group: radius for grouping of the predictions
        thresholds: `list` of score thresholds for cue point detection
        strategies: `list` of cluster center selection strategies
        fixed_groups: bool,
            `True`: groups defined by max distance to last element,
            `False`: groups defined by a max dist to next element
        result_dir: path to folder where results are saved
    """
    # load predictions from saved file
    all_predictions = load_predictions(result_dir)
    
    '''
    # create folder for result files
    result_dir = os.path.join(result_dir, 'score')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    '''
        
    cue_points = {}
    for i, pred in all_predictions.items():
        
        # prepare mapping
        results = {t: {s: [] for s in strategies} for t in thresholds}
        for t in thresholds:
            p = filter_score(pred, t)
            for s in strategies:
                if fixed_groups:
                    results[t][s] = find_cues_grouped(p, s, w_group, fixed_groups=True)
                else:
                    results[t][s] = find_cues_grouped(p, s, w_group)

        cue_points[i] = results

    return cue_points


def post_process_topk(topk, min_dist, result_dir):
    """
    Post process the predictions of a model and save the results. Filters predictions based on the top k candidates
    that lie at least a specified minimum distance apart.

    Args:
        model_name: name of the model or run to evaluate
        topk: int or list of how many candidates should be returned
        min_dist: radius for grouping of the predictions
        result_dir: path to folder where results are saved
    """

    # load predictions from saved file
    all_predictions = load_predictions(result_dir)
    
    # create folder for result files
    result_dir = os.path.join(result_dir, 'topk')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if isinstance(topk, int):
        topk = [topk]
    if isinstance(min_dist, int):
        min_dist = [min_dist]

    cue_points = {}
    for i, pred in all_predictions.items():
        # prepare mapping
        results = {k: {d: [] for d in min_dist} for k in topk}
        for k in topk:
            for d in min_dist:
                results[k][d] = find_cues_topk(pred, k, d)
        cue_points[i] = results

    return cue_points


def evaluate_results_(dist_tightness, model_name, image_dir, result_dir):
    filepath = os.path.join(result_dir, f'abl_{model_name}.json')
    assert os.path.exists(filepath), f'File {filepath} does not exist.'
    with open(filepath, 'r') as f:
        all_predictions = json.load(f)

    # Get ground truth cue points
    data = []
    for image_name, predictions in all_predictions.items():
        image_name = image_name + '.png'
        gt_cues = get_gt_cues(image_dir, image_name)

        # division by zero
        if len(gt_cues) == 0:
            continue
        
        # Get predicted cue points
        for first, pred in predictions.items():
            for second, pd_cues in pred.items():
                
                if len(pd_cues) == 0:
                    continue
                
                # get mapping of (idx, value) for all closest gt_cues
                mapping = [min(enumerate(gt_cues), key=lambda x: abs(x[1] - pred)) for pred in pd_cues]
                
                # always same length as pd_cues, never empty
                distances = [abs(t - m[1]) for t, m in zip(pd_cues, mapping)]
                closest = [m[0] for m in mapping]
                matches = [dist <= dist_tightness for dist in distances]
                matches = [dist for dist, is_match in zip(distances, matches) if is_match]

                data.append({
                    'name': f'{model_name}:{image_name[:-4]}:{first}:{second}',
                    'n_pred': len(pd_cues),
                    'n_hits': len(matches),
                    'n_gt': len(gt_cues),
                    'precision': len(matches) / len(pd_cues),
                    'recall': len(set(closest)) / len(gt_cues),
                    'mean_acc': np.mean(distances),
                    'median_acc': np.median(distances),
                    'mean_acc_hits': np.mean(matches),      # could be 'nan'
                    'median_acc_hits': np.median(matches),  # could be 'nan'
                })

    d = pd.DataFrame(data)
    d.to_csv(os.path.join(result_dir, f'eval_{model_name}.csv'))


def evaluate_results(dist_tightness, gt_mapping, all_results, phrase_len=-1):
    """
    Evaluate the results of a model with extended correctness based on phrasing.
     
    Args:
        dist_tightness: int, radius for a match of a cue point
        gt_mapping: dict mapping image names to ground truth cue points and beats
        all_results: prediction results to evaluate
        phrase_len: int, multiple of 4 (bars), if -1 phrasing not considered
        returns: list with evaluation results
    """
    if phrase_len > 0:
        assert phrase_len % 4 == 0, 'Phrasing must be a multiple of 4 (bars).'
    
    # need gt_mapping to get the phrasing
    eval_data = []
    for image_name, predictions in all_results.items():

        # Get ground truth cue points and length of a phrase
        gt_cues = gt_mapping[image_name]['cues']
        d_phrase = gt_mapping[image_name]['beat'] * 4 * phrase_len
        
        # No division by zero
        if len(gt_cues) == 0:
            continue

        # Get predicted cue points
        for first, pred in predictions.items():
            for second, pd_cues in pred.items():
                
                if len(pd_cues) == 0:
                    continue
                
                # get mapping of (gt_idx, gt_cue) for all closest gt_cues
                mapping = [min(enumerate(gt_cues), key=lambda x: abs(x[1] - pred)) for pred in pd_cues]
                
                # always same length as pd_cues, never empty
                distances = [abs(t - m[1]) for t, m in zip(pd_cues, mapping)]
                closest_gt = [m[0] for m in mapping]
                match_mask = [dist <= dist_tightness for dist in distances]
                matches = [dist for dist, is_match in zip(distances, match_mask) if is_match]

                # EVAL DATA
                results = {}
                results['name'] = f'{image_name}-{first}-{second}'
                results['n_pred'] = len(pd_cues)
                results['n_hits'] = len(matches)
                results['n_gt'] = len(gt_cues)
                results['precision'] = len(matches) / len(pd_cues)
                results['recall'] = len(set(closest_gt)) / len(gt_cues)
                results['acc_mean'] = np.mean(distances)
                results['acc_median'] = np.median(distances)

                if phrase_len > 0:
                    distances = [d % d_phrase for d in distances]
                    match_mask = [dist <= dist_tightness for dist in distances]
                    matches = [dist for dist, is_match in zip(distances, match_mask) if is_match]

                    # PHRASE EVAL DATA
                    results['ext_n_hits'] = len(matches)
                    results['ext_precision'] = len(matches) / len(pd_cues)
                    results['ext_acc_mean'] = np.mean(distances)
                    results['ext_acc_median'] = np.median(distances)

                eval_data.append(results)

    return eval_data


def draw_result(description, gt_cues, pd_cues, image_path, other_cues=None):
    """
    Plot a predictions onto its spectrogram image.

    Args:
        description: description added on top of image
        gt_cues: list of ground truth cue points
        pd_cues: list of predicted cue points
        image_path: path to spectrogram image
    """
    # Draw cues on spectrogram
    spectrogram = Image.open(image_path).convert('RGB')
    spectrogram = draw_cues(spectrogram, gt_cues, pd_cues, other_cues)

    # Plot as grid for better visibility
    segments = []
    n_segments = 6
    W, H = spectrogram.size
    for i in range(n_segments):
        l = i * (W // n_segments)
        r = l + (W // n_segments)
        segments.append(spectrogram.crop((l, 0, r, H)))
    result = plot_grid(segments, n_segments, 1)
    result = draw_header(result, description)

    return result
    

def draw_all_results(model_name, gt_mapping, all_results, image_dir, result_dir):
    """
    Args:
        model_name: name of the model or run to evaluate
        gt_mapping: dict mapping image names to ground truth cue points and beats
        all_results: prediction results to evaluate
        image_dir: path to folder with spectrograms
        result_dir: path to folder where results are saved
    """
    for image_name, predictions in all_results.items():

        image_path = os.path.join(image_dir, image_name + '.png')
        gt_cues = gt_mapping[image_name]['cues']

        for first, pred in predictions.items():
            for second, pd_cues in pred.items():
                # Gather predicted points
                if len(pd_cues) == 0:
                    continue

                description = get_description(model_name, len(gt_cues), len(pd_cues), first, second)
                result = draw_result(description, gt_cues, pd_cues, image_path)

                # Save images
                filepath = '{0}_{1}_{2}.jpg'.format(image_name, first, second)
                filepath = os.path.join(result_dir, filepath)
                result.save(filepath, quality=75, optimize=True)


'''
========================================
            Loading Data
========================================
'''
def load_predictions(result_dir):
    """ Expects `predictions.json` in the result directory. """
    filepath = os.path.normpath(os.path.join(result_dir, f'/predictions.json'))
    assert os.path.exists(filepath), f'File {filepath} does not exist.'
    with open(filepath, 'r') as f:
        return json.load(f)


def load_results(result_dir):
    """ Expects `results.json` in the result directory. """
    filepath = os.path.normpath(os.path.join(result_dir, f'/results.json'))
    assert os.path.exists(filepath), f'File {filepath} does not exist.'
    with open(filepath, 'r') as f:
        return json.load(f)


def load_ground_truth(gt_filepath):
    """ Expects path to json with ground truth. """
    assert os.path.exists(gt_filepath), f'File {gt_filepath} does not exist.'
    with open(gt_filepath, 'r') as f:
        return json.load(f)
    

def write_results(results, result_dir):
    """ Writes `results.json` in the result directory. """
    # save data for later
    filepath = os.path.normpath(os.path.join(result_dir, f'/results.json'))
    with open(filepath, 'w') as f:
        json.dump(results, f)


def get_gt_cues(image_dir, image_name):
    """ Expects `annotations.json` in the result directory. """
    # Get ground truth cue points
    with open(os.path.normpath(f'{image_dir}/annotations.json'), 'r') as f:
        annotations = json.load(f)

    # Get cue points
    image_id = next(image['id'] for image in annotations['images'] if image['file_name'] == image_name)
    positions = [ann['position'] for ann in annotations['annotations'] if ann['image_id'] == image_id]
    positions = [0 if c < 0 else c for c in positions]

    return positions


'''
========================================
           Helper Fuctions
========================================
'''

def filter_score(predictions, threshold):
    # Filter the predictions on one track based on the score threshold
    scores = []
    positions = []
    for s, p in zip(predictions['scores'], predictions['positions']):
        if s >= threshold:
            scores.append(s)
            positions.append(p)

    return {
        'scores': scores,
        'positions': positions,
    }


def group_by_distance(frames: list[int], w: int):
    """ 
    Group `frames` into clusters that are `w` apart. Clusters are returned as nested lists.
    
    Args:
        frames: required to be sorted
        w: min. distance between clusters
    """
    assert len(frames) > 1, 'No frames to group'

    prev = None
    cluster = []
    for f in frames:
        if prev is None or f - prev <= w:
            cluster.append(f)
        else:
            yield cluster
            cluster = [f]
        prev = f
    if cluster:
        yield cluster


def group_fixed_size(frames: list[int], w: int):
    """ 
    Group `frames` into clusters that are `w` wide. Clusters are returned as nested lists.
    
    Args:
        frames: required to be sorted
        w: width of clusters
    """
    assert len(frames) > 1, 'No frames to group'

    prev = frames[0]
    cluster = [prev]

    for f in frames[1:]:
        if f - prev <= w:
            cluster.append(f)
        else:
            yield cluster
            cluster = [f]
            prev = f
    if cluster:
        yield cluster


def find_cues_grouped(predictions, strategy, w_group, fixed_groups=False):
    """ 
    Returns a list of best cue frames based based on the selected grouping strategy and group width.

    Args:
        predictions: list of dicts with keys: 'labels', 'scores'
        strategy: ('mean', 'median', 'best', or 'weighted') clustering strategy to use
        w_cluster: window size to group the predicted cue frames
    """

    # score will have same order as grouped frames elements
    scores = predictions['scores']
    positions = predictions['positions']
    assert len(scores) == len(positions), 'Scores and positions must be the same length'

    # exit early if no predictions or just one
    if len(positions) <= 1:
        return positions

    # sort in parallel based on positions
    positions, scores = zip(*sorted(zip(positions, scores)))
    if fixed_groups:
        clusters = list(group_fixed_size(positions, w_group))
    else:
        clusters = list(group_by_distance(positions, w_group))

    match strategy:
        case 'mean':
            return [int(sum(c) / len(c)) for c in clusters]
        case 'median':
            return [int(np.median(c)) for c in clusters]
        case 'best':
            i = 0
            best = []
            for c in clusters:
                n = len(c)
                best.append(int(c[np.argmax(scores[i:i+n])]))
                i += n
            return best
        case 'weighted':
            i = 0
            avg = []
            for c in clusters:
                n = len(c)
                avg.append(int(sum(c + list(scores[i:i+n])) / n))
                i += n
            return avg
        case _:
            raise ValueError(f'Invalid clustering strategy: {strategy}')


def find_cues_topk(predictions, k, min_dist):
    """ 
    Returns a list of the top k cue predictions that lie at least the min distance apart.

    Args:
        predictions: list of dicts with keys: 'labels', 'scores'
        k: how many cues to return
        min_dist: radius for which no other top candidate can be selected
    """
    # score will have same order as grouped frames elements
    scores = predictions['scores']
    positions = predictions['positions']
    assert len(scores) == len(positions), 'Scores and positions must be the same length'

    # exit early if no predictions or just one
    if len(positions) <= 1:
        return positions

    # Sort scores in descending order, make sure positions have same permtations
    scores, positions = zip(*sorted(zip(scores, positions), reverse=True))
    scores, positions = iter(scores), iter(positions)

    top_cues = []
    while k > 0:
        p = next(positions, None)
        s = next(scores, None)
        if p is None or s is None:
            break
        if not any(abs(c - p) < min_dist for c in top_cues):
            top_cues.append(p)
            k -= 1
    
    return top_cues


def find_cues_peak(predictions, score_threshold, radius):
    """
    Aggregate predictions to that are close together.
    
    Args:
        predictions: list of dicts with keys: 'positions', 'scores'
        score_threshold: min score of a prediction to be considered for a peak
        radius: min distance between peaks
    
    returns: list of peak scores and peak positions
    """
    # Sort by position
    scores = predictions['scores']
    positions = predictions['positions']
    positions, scores = zip(*sorted(zip(positions, scores)))

    # Find peaks
    peak_idx, _ = find_peaks(scores, height=score_threshold, distance=radius)
    peak_scores = [scores[idx] for idx in peak_idx]
    peak_positions = [positions[idx] for idx in peak_idx]

    return peak_scores, peak_positions


def get_box_window_idx(target: list):
    """ Returns the index of the windows that contains boxes """
    return [i for i, t in enumerate(target) if t['boxes'].shape[0] > 0]


def find_phrase_boundaries(cue_dict, phrase_len):
    """
    Returns all phrase boundaries based on the provided cue points. The phrases are `phrase_len` apart.
    In case `phrase_len` is 0, a copy of passed cue points is returned.
    
    Args:
        cue_dict: dict, expected keys: 'cues': List[int], 'bpm': float, 'duration': int
        phrase_len: int, length of a phrase in bars
    """

    gt_cues = cue_dict['cues'].copy()
    if phrase_len <= 0:
        return gt_cues
    
    dur = cue_dict['duration']
    
    # first get durations in seconds
    bar = 240/cue_dict['bpm']
    phrase = phrase_len * bar

    # ...then convert to frames (avoids rounding errors)
    phrase, bar = librosa.time_to_frames([phrase, bar])
   
    if len(gt_cues) < 1:
        return [i * phrase for i in range(dur//phrase)]

    i = 0
    gt_cues.sort()
    curr = gt_cues[i]
    boundaries = [curr]
    
    # add boundaries at start if needed
    while curr - phrase >= 0:
        curr -= phrase
        boundaries.append(curr)
    
    # reset
    curr = gt_cues[i]
    while curr + phrase <= dur:
        curr += phrase
        # phrase skipping or landing on a cue, take that instead
        if i < len(gt_cues) - 1 and curr + bar > gt_cues[i+1]:
            i += 1
            curr = gt_cues[i]
        boundaries.append(curr)
    
    return boundaries


def get_match_mask(pd_positions, gt_positions, tightness):
    
    # maps pd_positions to (gt_idx, gt_val)
    dist_mapping = [min(enumerate(gt_positions), key=lambda x: abs(x[1] - pred)) for pred in pd_positions]
    matches = [abs(pd_pos - gt_pos[1]) <= tightness for pd_pos, gt_pos in zip(pd_positions, dist_mapping)]
    
    return matches


def get_match_mask_fn(pd_positions, gt_positions, tightness, t_phrase):
    """
    Returns a list of booleans indicating if a predicted cue is a match to a ground truth cue.
    For each missed cue, False is appended to the list.
    """
    dist_mapping = [min(enumerate(gt_positions), key=lambda x: abs(x[1] - pred)) for pred in pd_positions]
    distances = [abs(pd_pos - gt_pos[1]) for pd_pos, gt_pos in zip(pd_positions, dist_mapping)]
    
    # if t_phrase is 0, we have to make the modulo ineffective so only actual cues are used
    t_phrase = max(distances) + 1 if t_phrase == 0 else t_phrase
    matches = [d % t_phrase <= tightness for d in distances]
    found_cues = [dm[0] for dm, is_match in zip(dist_mapping, matches) if is_match]
    
    # extend with false for missed cues
    num_missed = len(gt_positions) - len(found_cues)
    # matches.extend([False] * missed)

    return matches, num_missed


def get_proba_labels(predictions, ground_truth, phrase_len, radius, beat=None):
    """
    Return a list with probabilities and true-false mask for all predicted cues.

    Args:
        predictions: dict of predictions for all tracks
        ground_truth: dict of ground truth for all tracks
        phrase_len: int, length of a phrase in bars
        radius: int, min dist to next possible peak
        beat: str, 'one', 'half', 'quarter' for tightness, default: 0.5s
    """
    labels = []
    probas = []
    tightness = librosa.time_to_frames(0.25)
    for track_id, prediction in predictions.items():
        # find the predictions based on the threshold
        scores, positions = find_cues_peak(prediction, 0, radius)
        if len(positions) == 0:
            continue
        
        # find the matches based on the phrase length
        # actual = find_phrase_boundaries(ground_truth[track_id], phrase_len)
        actual = ground_truth[track_id]['cues']
        match beat:
            # absolute distance -> only use half of tightness
            case 'one':
                tightness = librosa.time_to_frames(60/(2*ground_truth[track_id]['bpm']))
            case 'half':
                tightness = librosa.time_to_frames(60/(4*ground_truth[track_id]['bpm']))
            case 'quarter':
                tightness = librosa.time_to_frames(60/(8*ground_truth[track_id]['bpm']))
            case _:
                pass
  
        matches, n_missed = get_match_mask_fn(positions, actual, tightness, phrase_len)
        if n_missed > 0:
            # add the missed predictions with confidence 0 to indicate it was not guessed
            scores.extend([0.0] * n_missed)
            matches.extend([False] * n_missed)
        probas.extend(scores)
        labels.extend(matches)

    return probas, labels




'''
==========================================
              MAIN FUNCTION
==========================================
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate cue point detection model.')

    # Model params
    parser.add_argument('--model', nargs='+', type=str, help="Model name(s) to evaluate.")
    parser.add_argument('--model_dir', type=str, help="Directory with model checkpoint(s) to evaluate.")
    parser.add_argument('--checkpoint', type=str, help="Model checkpoint.")
    parser.add_argument('--image_processor', type=str, help="HF Image processor.")

    # Data params
    parser.add_argument('--image_dir', type=str, help="Directory with test images.")
    parser.add_argument('--result_dir', type=str, help="Directory for results file.")
    parser.add_argument('--gt_filepath',type=str, help="Path to ground truth annotations file.")

    # Eval params
    parser.add_argument('--box_width', type=int, nargs='*', help="Width of box around cue point.")
    parser.add_argument('--window_width', type=int, help="Width of spectrograms windows.")
    parser.add_argument('--overlap', type=float, default=0.5, help="Overlap of the spectrograms windows.")
    parser.add_argument('--dist_tightness', type=float, default=0.5, help="Tightness for a match of a cue point.")
    parser.add_argument('--phrase_len', type=int, nargs='*', help="Length of a phrase in bars (multiple of 4), use -1 to idicate ground truth only.")
    parser.add_argument('--save_as', type=str, required=True, help="Name of the csv file to save the results to.")

    # Ablation params
    parser.add_argument('--thresholds', type=float, nargs='*', help="Score thresholds for cue point detection.")
    parser.add_argument('--strategies', type=str, choices=['mean', 'median', 'best', 'weighted'], nargs='*', help="Cluster center selection strategies.")
    parser.add_argument('--group_width', type=int, default=20, help="Width of cue point groups/clusters.")
    parser.add_argument('--topk', type=int, nargs='*', help="Number of top candidates to return.")
    parser.add_argument('--min_dist', type=int, nargs='*', help="Min distance between top candidates.")

    args = parser.parse_args()
    
    scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) - 10e-10)

    if args.box_width is not None:
        assert len(args.box_width) == len(args.model), 'Box widths must match the models or omit box width to use default width 7 for all models.'

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    for i, model in enumerate(args.model):
        # Set up an eval sub folder for the run if not already existing
        eval_dir = os.path.join(args.result_dir, model)
        if not os.path.isdir(eval_dir):
            os.mkdir(eval_dir)
        
        # Load predictions or make new ones
        pred_file = os.path.join(eval_dir, f'predictions_{int(args.overlap * 100)}.json')
        if os.path.exists(pred_file):
            print(f"Predictions for model {model} already exist.")
            # print(f"Loading predictions for model {model}")
            # with open(pred_file, 'r') as f:
            #     predictions = json.load(f)
        else:
            assert os.path.exists(args.image_dir), f'Image folder {args.image_dir} does not exist.'
            assert os.path.exists(os.path.join(args.model_dir, model)), f'Model {model} does not exist.'
            
            message = f"Predictions for model {model} with"
            if args.box_width is not None:
                w = args.box_width[i]
                message += f"box width {w} px"
            else:
                message += f"default box width (7 px)"
            print(message)
            
            predictions = predict_all(args, model, w)
            with open(pred_file, 'w') as f:
                json.dump(predictions, f)

    print('evaluating...')
    # Look for best parameters
    # ap_values = []
    # thresholds = np.linspace(0.1, 0.9, num=10)
    gt_map = load_ground_truth(args.gt_filepath)
    radius = list(librosa.time_to_frames([i * 240/174 for i in [16, 8]]))
    if args.phrase_len is None:
        args.phrase_len = [0, 16, 8, 4]

    # if isinstance(args.dist_tightness, float):
        # args.dist_tightness = librosa.time_to_frames(args.dist_tightness)

    # 1. load model data
    M = len(args.model)
    D = 4 # distance tightness measures
    P = len(args.phrase_len)
    R = len(radius)

    AP_results = np.zeros((M*D, P*R))
    columns = ['' for _ in range(P*R)]
    indices = ['' for _ in range(M*D)]
    for m in range(M):
        model = args.model[m]
        print(f'evaluating model {model}')
        # load predictions
        with open(os.path.join(args.result_dir, model, f'predictions_{(int(args.overlap * 100))}.json'), 'r') as f:
            predictions = json.load(f)

        # AP for radius 1 and 2
        m *= D
        for p, p_len in enumerate(args.phrase_len):
            for r, rad in enumerate(radius):
                i = r + p*R
                columns[i] = f'AP:{p_len}_r{r}'
                probas, mapping = get_proba_labels(predictions, gt_map, p_len, rad, beat='one')
                AP_results[m, i] = average_precision_score(mapping, scale(probas))
                indices[m] = f'{model}_one'

                probas, mapping = get_proba_labels(predictions, gt_map, p_len, rad, beat='half')
                AP_results[m+1, i] = average_precision_score(mapping, scale(probas))
                indices[m+1] = f'{model}_half'

                probas, mapping = get_proba_labels(predictions, gt_map, p_len, rad, beat='quarter')
                AP_results[m+2, i] = average_precision_score(mapping, scale(probas))
                indices[m+2] = f'{model}_quarter'

                probas, mapping = get_proba_labels(predictions, gt_map, p_len, rad, args.dist_tightness)
                AP_results[m+3, i] = average_precision_score(mapping, scale(probas))
                indices[m+3] = f'{model}_{args.dist_tightness}'

    df = pd.DataFrame(AP_results, columns=columns, index=indices)
    df.to_csv(os.path.join(args.result_dir, f'{args.save_as}.csv'))

    quit()

    ap_vals = np.zeros((len(thresholds), len(radius)))
    p_vals = np.zeros((len(thresholds), len(radius)))
    r_vals = np.zeros((len(thresholds), len(radius)))

    for i, t in enumerate(thresholds):
        for j, r in enumerate(radius):
            probas = []
            labels = []
            for id, pred in predictions.items():
                pred['scores'] = scale(pred['scores'])
                scores, positions = find_cues_peak(pred, t, r)
                cue_positions = find_phrase_boundaries(gt_map[str(id)], 0)
                matches = get_match_mask(positions, cue_positions, args.dist_tightness)
                probas.extend(scores)
                labels.extend(matches)

            pre, rec, _ = precision_recall_curve(labels, scale(probas))
            ap = average_precision_score(labels, scale(probas))
            ap_vals[i, j] = ap
            p_vals[i, j] = np.mean(pre)
            r_vals[i, j] = np.mean(rec)

    
    print('saving results...')
    save_as_csv(thresholds, radius, ap_vals, p_vals, r_vals, os.path.join(args.result_dir, 'abl'))

    # make_scatter_plot(ap_vals, thresholds, radius, f'AP {args.model}', args.result_dir)
    # make_scatter_plot(p_vals, thresholds, radius, f'Pre {args.model}', args.result_dir)
    # make_scatter_plot(r_vals, thresholds, radius, f'Rec {args.model}', args.result_dir)


    '''
    if args.ablation_topk:
        assert args.topk is not None, 'Please specify at least one topk value.'
        assert args.min_dist is not None, 'Please specify at least one min_dist value.'
        print(f"Ablation for model {args.model} with topk {args.topk} and min_dist {args.min_dist}")
        post_process_topk(args.model, args.topk, args.min_dist, args.result_dir)

    if args.ablation_score or args.ablation_score_fixed:
        assert args.thresholds is not None, 'Please specify at least one threshold value.'
        assert args.strategies is not None, 'Please specify at least one strategy (choices: mean, median, best, weighted).'
        print(f"Ablation for model {args.model} with thresholds {args.thresholds} and strategies {args.strategies}")
        if args.ablation_score_fixed:
            post_process_score(args.model, args.group_width, args.thresholds, args.strategies, True, args.result_dir)
        else:
            post_process_score(args.model, args.group_width, args.thresholds, args.strategies, args.result_dir)
    
    if args.draw_predictions:
        print(f"Drawing predictions for model {args.model}")
        draw_all_results(args.model, args.image_dir, args.result_dir)

    if args.eval:
        print(f"Evaluating predictions for model {args.model}")
        evaluate_results(args.dist_tightness, args.model, args.image_dir, args.result_dir)
    '''

