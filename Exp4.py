import cv2 as cv
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border,watershed
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_local
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from copy import copy
import numpy as np
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import logging

def ca_prep(config):
    logger = logging.getLogger('ca_prep')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)

    default_config = {
        'working_dir': None,
        'ca_filename': None,
        'timeline_filename': None,
        'stimulus_filename': None,
        'binsize': 60,
        'stepsize': 40,
        'show_motion_correction_result': True,
        'segmentation_params': {
            'hpfiltSig': .1,
            'localThreKerSize': 9,
            'smoothSig': 3,
            'binaryThre': .5,
            'minSizeLim': 20,
            'maxSizeLim': 500,
            'bgKerSize': 2,
            'fgKerSize': 1
        },
        'show_segmentation_result': True,
        'display_traces': True,
        'save_h5': True,
    }

    default_config.update(config)
    config = default_config
    required_field_name = ['working_dir', 'ca_filename', 'timeline_filename', 'stimulus_filename']
    if all([config[i] is not None for i in required_field_name]):
        logger.info('Loading Ca movie')
        ca_movie = load_ca_movie(config['working_dir'] + config['ca_filename'])
        if config['show_motion_correction_result']:
            logger.info('Start motion correction, corrected std image will be shown')
        else:
            logger.info('Start motion correction, corrected std image will NOT be shown')
        reg_frames, std_image = motion_correction(ca_movie, binsize=config['binsize'], stepsize=config['stepsize'],show_result=config['show_motion_correction_result'])  # 10, 10
        if config['show_segmentation_result']:
            logger.info('Running auto ROI detection, segmentation result will be shown')
        else:
            logger.info('Running auto ROI detection, segmentation result will NOT be shown')
        roi_mask = cell_segmentation(std_image, config['segmentation_params'],show_segmentation_result=config['show_segmentation_result'])
        if config['display_traces']:
            logger.info('Extracting calcium traces from Ca movie, extracted Ca traces will be plotted')
        else:
            logger.info('Extracting calcium traces from Ca movie, extracted Ca traces will NOT be plotted')
        raw_ca_traces = extract_calcium_signals(roi_mask, reg_frames, display_traces=config['display_traces'])
        logger.info('Aligning stimulus parameter to ca traces')
        stim_array = align_stimulus_to_ca_frames(timefn=config['working_dir'] + config['timeline_filename'], stim_fn=config['working_dir'] + config['stimulus_filename'])
        roi_id = np.unique(roi_mask)[1:]
        formatted = pd.DataFrame(np.array(raw_ca_traces).T, columns=roi_id)
        for k, v in stim_array.items():
            formatted["stim_" + k] = v
        if config['save_h5']:
            save_fn = config['ca_filename'].split('.')[0]
            save_full_path = config['working_dir']+"processed_"+save_fn+'.h5'
            h5io = h5.File(save_full_path,'w')
            h5io.create_dataset('STD image',data=std_image)
            h5io.create_dataset('ROI mask',data=roi_mask)
            h5io.close()
            formatted.to_hdf(save_full_path,'stim_ca_traces')
            logger.info('Save processed data into h5 files: {}'.format(save_full_path))
        logger.info('Preprocessing finished')
        return formatted
    else:
        logger.critical('At least one of the following fields are not set: {}'.format(required_field_name))
        return None

def load_ca_movie(fn):
    return np.array(io.imread(fn))

# function for motion correction (registration) of calcium frames (tiff-stack)
def motion_correction(raw_frames, binsize=500, stepsize=250, show_result=True):
    if binsize is not None:
        binsize = binsize
    if stepsize is not None:
        stepsize = stepsize
    image_template = np.mean(raw_frames[:binsize], axis=0)
    i = stepsize
    regTifImg = copy(raw_frames)
    old_drift = None
    while i < raw_frames.shape[0]:
        moving_template = np.mean(raw_frames[i:i + binsize], axis=0)
        image_drift = phase_cross_correlation(image_template, moving_template)[0]
        if old_drift is None:
            old_drift = image_drift
        for o in range(min(stepsize, raw_frames.shape[0] - i)):
            itershift = np.vstack(
                [np.eye(2), image_drift[::-1] * (1 - o / stepsize) + old_drift[::-1] * o / stepsize]).T
            regTifImg[i + o] = cv.warpAffine(raw_frames[i + o], itershift, tuple(image_template.shape))
        old_drift = image_drift
        i += stepsize
    reg_frames = regTifImg
    std_image = np.std(reg_frames, axis=0)
    if show_result:
        fig_name = 'Registration_STD_image'
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), num=fig_name)
        ax[0].set_title('Raw STD image')
        ax[0].imshow(np.std(raw_frames, axis=0))
        ax[1].set_title('Registered STD image')
        ax[1].imshow(std_image)
        fig.tight_layout()
        plt.show()
    return reg_frames, std_image


rnorm = lambda x: (x - x.min()) / (x.max() - x.min())


# function for cell segmentation based on watershed algorithm
def cell_segmentation(std_image, segmentation_params_arg=None,show_segmentation_result=True):
    segmentation_params = {
        'hpfiltSig': .1,
        'localThreKerSize': 25,
        'smoothSig': 3,
        'binaryThre': .5,
        'minSizeLim': 50,
        'maxSizeLim': 500,
        'bgKerSize': 2,
        'fgKerSize': 1
    }

    if segmentation_params_arg is not None:
        segmentation_params.update(segmentation_params_arg)

    # image smoothing
    hpRaw = cv.GaussianBlur(std_image, (0, 0), segmentation_params['hpfiltSig'])
    bpRaw = hpRaw / threshold_local(hpRaw, segmentation_params['localThreKerSize'])
    smoothed_std_image = cv.GaussianBlur(rnorm(bpRaw), (0, 0), segmentation_params['smoothSig'])

    # image binarization
    bwIm = (smoothed_std_image > (
            np.mean(smoothed_std_image) + segmentation_params['binaryThre'] * np.std(smoothed_std_image)))
    binarized_std_image = remove_small_objects(bwIm, segmentation_params['minSizeLim'], connectivity=4).astype(np.uint8)

    # compute bgMarker and fgMarker
    bgDilateKer = np.ones((segmentation_params['bgKerSize'],) * 2, np.uint8)
    bgMarker = clear_border(cv.dilate(binarized_std_image, bgDilateKer, 1) > 0)
    conn = np.ones((3, 3,))
    fgDilateKer = np.ones((segmentation_params['fgKerSize'],) * 2, np.uint8)
    maxCoord = peak_local_max(smoothed_std_image, footprint=conn, indices=False, exclude_border=0)
    fgMarker = clear_border(cv.dilate(maxCoord.astype(np.uint8), fgDilateKer)) > 0

    # watershed
    distanceMap = ndi.distance_transform_edt(bgMarker)
    markers = ndi.label(fgMarker)[0]
    raw_labels = watershed(-distanceMap, markers, mask=bgMarker, watershed_line=True)

    # filter ROIs by size
    temp_val, temp_idx = np.unique(raw_labels, return_inverse=True)
    num_of_val = np.bincount(temp_idx)
    exclude_val = temp_val[np.bitwise_or(num_of_val <= np.array(segmentation_params['minSizeLim']),
                                         num_of_val >= np.array(segmentation_params['maxSizeLim']))]
    labels = copy(raw_labels)
    labels[np.isin(raw_labels, exclude_val)] = 0

    # order labels
    sorted_labels = copy(labels)
    for i, v in enumerate(np.unique(labels)):
        sorted_labels[labels == v] = i

    roi_mask = sorted_labels

    if show_segmentation_result:
        fig_name = 'sizeFilter'
        fig = plt.figure(figsize=(16, 8), num=fig_name)
        plt.title('ROI Map')
        plt.imshow(std_image / 40 + 40 * np.squeeze(np.abs(np.gradient(sorted_labels)).sum(axis=0) > 0))
        fig.tight_layout()
        plt.show()

    return roi_mask


# function to extract single calcium traces based on segmented ROI mask
def extract_calcium_signals(roi_mask, reg_frames, display_traces=True):
    raw_ca_traces = []
    for i in np.unique(roi_mask):
        if i > 0:
            raw_ca_traces.append(reg_frames[:, roi_mask == i].sum(axis=1))

    if display_traces:
        fig_name = 'Calcium Traces'
        fig = plt.figure(figsize=(16, 8), num=fig_name)
        plt.title('Raster Plot ROIs')
        plt.imshow(np.array(raw_ca_traces))
        plt.xlabel('frames')
        plt.ylabel('ROIs')
        fig.tight_layout()
        plt.show()
    return raw_ca_traces


# function to calculate dff
def calc_dff(raw_ca_traces, num_frames_avg=25, display_traces=True):
    dff_ca_traces = []
    for i in range(len(raw_ca_traces)):
        sorted_trace = np.sort(raw_ca_traces[i])
        dff_ca_traces.append(
            (raw_ca_traces[i] - np.mean(sorted_trace[1:num_frames_avg])) / np.mean(sorted_trace[1:num_frames_avg]))
    if display_traces:
        fig_name = 'DFF Calcium Traces'
        fig = plt.figure(figsize=(16, 8), num=fig_name)
        plt.title('Raster Plot ROIs DFF')
        plt.imshow(np.array(dff_ca_traces))
        plt.xlabel('frames')
        plt.ylabel('ROIs')
        fig.tight_layout()
        plt.show()
    return dff_ca_traces

def frame_detection(mirTime, mirPos):
    diffMirPos = np.diff(mirPos)
    temp_ind, _ = find_peaks(diffMirPos, prominence=np.std(diffMirPos))
    peak_ind, _ = find_peaks(mirPos[temp_ind], prominence=np.std(mirPos))
    peak_ind = temp_ind[peak_ind]
    valley_ind, _ = find_peaks(-mirPos[temp_ind], prominence=np.std(mirPos))
    valley_ind = temp_ind[valley_ind]
    ca_frame_idx = np.unique(np.concatenate([valley_ind, peak_ind], axis=0)) - 1
    ca_frame_time = mirTime[ca_frame_idx]
    return ca_frame_time, ca_frame_idx

def extract_timeinfo(timefn):
    timefile = h5.File(timefn)
    mirTime = timefile['ai_y_mirror_in_time']
    mirPos = timefile['ai_y_mirror_in']
    mirTime = np.squeeze(mirTime)
    mirPos = np.squeeze(mirPos)
    ca_frame_time, ca_frame_idx = frame_detection(mirTime,mirPos)
    timefile.close()
    return ca_frame_time, ca_frame_idx

def align_stimulus_to_ca_frames(stim_fn,timefn):
    ca_frame_time,ca_frame_idx = extract_timeinfo(timefn)
    stim_file = h5.File(stim_fn)
    phaseAttr = [(int(k.strip('phase')),v.attrs)for k,v in stim_file.items() if 'phase' in k]
    phaseIdx = np.argsort([i[0] for i in phaseAttr])
    phaseAttr = [phaseAttr[i] for i in phaseIdx]
    stim_array = np.zeros([ca_frame_idx.shape[0],3])
    for i in phaseAttr:
        phase_num = i[0]
        ca_start_frame = np.argmin(np.abs(i[1]['__start_time'] - ca_frame_time))
        ca_end_frame = np.argmin(np.abs(i[1]['__end_time'] - ca_frame_time))+1
        stim_array[ca_start_frame:ca_end_frame,0] = phase_num
        if 'angular_velocity' in i[1].keys():
            stim_array[ca_start_frame:ca_end_frame,1] = i[1]['angular_velocity']
            stim_array[ca_start_frame:ca_end_frame,2] = i[1]['angular_period']
        else:
            stim_array[ca_start_frame:ca_end_frame, 1] = -99
            stim_array[ca_start_frame:ca_end_frame, 2] = -99
    stimulus_info = {'phase': stim_array[:,0],
                     'speed': stim_array[:,1],
                     'spatial_freq': stim_array[:,2],
                     'time': ca_frame_time,
                     }
    stimulus_info = pd.DataFrame(data=stimulus_info)
    stimulus_info.set_index('time')
    return stimulus_info
