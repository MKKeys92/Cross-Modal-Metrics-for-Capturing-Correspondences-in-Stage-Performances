import math

import numpy as np
import os, sys, librosa
from scipy.signal import find_peaks

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import mir_eval
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from RunScriptsAndConfigs.Preprocessing.Audio.extract_audio_features import get_audio_extraction_conf
from Sources.utils.util import logger

cmap = 'gray_r'
class SSMCalculator:
    def __init__(self, model_config, runtime_args, plot, plot_path):
        self.model_config = model_config
        self.L_light_Kernel = 31
        self.L_audio_Kernel = 31
        self.L_light_Sm = 81
        self.L_audio_Sm = 81
        self.H_light = 10
        self.H_audio = 10
        self.peaks_distance = 15
        self.peaks_prominence = 0.04
        self.framerate = 30
        self.mir_segment_window = 2
        self.plot = plot
        self.plot_path = plot_path
        self.shift_val = 3 #shifting SSM 3 Frames in every direction
        self.audio_extraction_config = get_audio_extraction_conf(self.model_config.prepro_config['audio_extraction_config'])

    def set_mir_segment_window(self, val):
        self.mir_segment_window = val

    def set_novelty_kernel_size(self, val):
        self.L_audio_Kernel = val
        self.L_light_Kernel = val

    def set_smoothing_filter_size(self, val):
        self.L_audio_Sm =val
        self.L_light_Sm =val


    def set_downsampling_size(self, val):
        self.H_audio = val
        self.H_light = val

    def get_trimmed_novelty_peaks(self, novelty, kernelsize):
        # trim beginning and End
        # edge_trim = np.ones(shape=novelty.shape)
        #
        # k2 = int(kernelsize/2)
        #
        # edge_trim[:k2] = 0
        # edge_trim[-k2:] = 0
        # edge_trim[k2:kernelsize] = np.linspace(0, 1, k2)
        # edge_trim[- kernelsize: -k2] = np.linspace(1, 0, k2)
        nov = novelty #* edge_trim

        peaks, _ = find_peaks(nov, distance=self.peaks_distance, prominence=self.peaks_prominence)
        mask = np.ones(len(peaks), dtype=bool)
        # for idx in range(len(peaks)):
        #     if peaks[idx] <= k2 + self.peaks_distance or peaks[idx] >= len(
        #             nov) - k2 - self.peaks_distance:
        #         mask[idx] = False
        peaks = peaks[mask]

        return peaks, nov

    def create_time_boundaries(self, peaks, seq_len, H):
        if peaks.size == 0:
            return np.transpose(np.array(([0], [seq_len * 1 / self.framerate*H])))
        else:
            b_start = peaks * 1 / self.framerate * H
            b_end = b_start
            if peaks[0] != 0:
                b_start = np.append([0], b_start)
            else:
                b_end = b_end[1:]

            if peaks[-1] != seq_len:
                b_end = np.append(b_end, [(seq_len * 1 / self.framerate*H)])
            else:
                b_start = b_start[:,-2]

            return np.transpose(np.array((b_start, b_end)))

    def eval_all_SSM_metrics(self, light_feature_seq, file_name, chroma_stft):

        res = []
        SSM_Corrs = []

        sv = range(-self.shift_val, self.shift_val + 1)
        for s in sv:
            ax = None
            if self.plot and s == 0:
                fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [2, 0.5,2,0.5]}, figsize=(5,15))
                #fig.tight_layout()

            light_shifted = light_feature_seq
            if s >0:
                light_shifted = light_feature_seq[:-s,:]
            elif s < 0:
                light_shifted = light_feature_seq[-s:,:]

            l, S_l, nov_l = self.calc_light_boundaries_and_SSM(light_shifted, ax)
            a, S_a, nov_a = self.calc_audio_boundaries_and_SSM(chroma_stft, ax, s)

            if self.plot and s == 0:
                p = os.path.join(self.plot_path,file_name + '.png' )
                plt.savefig(p)
                plt.close(fig)

            #SSM_Distance = np.sum(np.abs(S_l - S_a)) / S_l.size

            SSM_Corr, _ = pearsonr(S_l.flatten(), S_a.flatten())
            max_ks = max(self.L_audio_Kernel, self.L_light_Kernel)

            Nov_Corr = 0
            if 2*max_ks >= len(nov_a) or len(nov_a[max_ks: -max_ks])<2 or len(nov_l[max_ks: -max_ks])<2:
                logger.log("Kernel size bigger than light / audio sample for file " + file_name)
            else:
                Nov_Corr, _ = pearsonr(nov_a[max_ks: -max_ks], nov_l[max_ks: -max_ks])

            p, re, f = mir_eval.segment.detection(l,a, self.mir_segment_window)

            SSM_Corrs.append(SSM_Corr)
            res.append((p, re, f, SSM_Corr, Nov_Corr))

        max_value = max(SSM_Corrs)
        max_index = SSM_Corrs.index(max_value)

        return  res[max_index]

    def calc_light_boundaries_and_SSM(self, light_feature_seq, ax):
        nov, S = self.calc_light_novelty(light_feature_seq)
        peaks, trimmed_nov = self.get_trimmed_novelty_peaks(nov, self.L_light_Kernel)
        if ax is not None:
            T_coef = np.arange(trimmed_nov.shape[0]) / 3
            libfmp.b.plot_matrix(S, Fs=1, cmap=cmap, title="Lighting",
                                 ylabel='Time (seconds)', xlabel="", colorbar=True, ax = [ax[0]], T_coef=T_coef, F_coef=T_coef)
            fig, _ax, line = libfmp.b.plot_signal(trimmed_nov, Fs=3, color='k', ylabel="Novelty", ax = ax[1], xlabel='' )
            viz_peaks = T_coef[peaks]
            #ax[1][0].vlines(x=viz_peaks,  ymin=0, ymax=1, label='Light Peaks', ls='--', colors=['lime'])
            ax[1].set_ylim([0.00, 0.25])

        return self.create_time_boundaries(peaks, len(trimmed_nov), self.H_light), S, trimmed_nov

    def calc_audio_boundaries_and_SSM(self, chroma_stft, ax, shiftvalue):
        nov, S = self.calc_audio_novelty(chroma_stft, shiftvalue)
        peaks, trimmed_nov = self.get_trimmed_novelty_peaks(nov, self.L_audio_Kernel)
        if ax is not None:
            T_coef = np.arange(trimmed_nov.shape[0]) / 3
            libfmp.b.plot_matrix(S, Fs=1, cmap=cmap, xlabel="", ylabel='Time (seconds)', title = "Audio",
                                 colorbar=True, ax = [ax[2]], T_coef=T_coef, F_coef=T_coef)
            fig, _ax, line = libfmp.b.plot_signal(trimmed_nov, Fs=3, color='k', ax = ax[3], ylabel="Novelty")
            viz_peaks = T_coef[peaks]
            #ax[1][1].vlines(x=viz_peaks, ymin=0, ymax=1, label='Audio Peaks', ls='--', colors=['lime'])
            ax[3].set_ylim([0.00, 0.25])

        return self.create_time_boundaries(peaks, len(trimmed_nov), self.H_audio), S, trimmed_nov

    def calc_light_novelty(self, light_feature_seq):
        X, Fs_X, S = self.compute_sm_from_light_features(light_feature_seq, L=self.L_light_Sm, H=self.H_light, L_smooth=1, thresh=1)

        nov = self.compute_novelty_ssm(S, L=self.L_light_Kernel, exclude=True)

        return nov, S

    def calc_audio_novelty(self, chroma_stft, shiftvalue):
        x_duration, X, Fs_X, S, I = self.compute_sm_from_chroma_stft(chroma_stft,
                                                                     L=self.L_audio_Sm, H=self.H_audio, L_smooth=1, thresh=1, shift = shiftvalue)
        nov = self.compute_novelty_ssm(S, L=self.L_audio_Kernel, exclude=True)
        return nov, S

    def compute_kernel_checkerboard_box(self, L):
        """Compute box-like checkerboard kernel [FMP, Section 4.4.1]

        Notebook: C4/C4S4_NoveltySegmentation.ipynb

        Args:
            L (int): Parameter specifying the kernel size 2*L+1

        Returns:
            kernel (np.ndarray): Kernel matrix of size (2*L+1) x (2*L+1)
        """
        axis = np.arange(-L, L + 1)
        kernel = np.outer(np.sign(axis), np.sign(axis))
        return kernel

    def compute_kernel_checkerboard_gaussian(self, L, var=1, normalize=True):
        """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
        See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

        Notebook: C4/C4S4_NoveltySegmentation.ipynb

        Args:
            L (int): Parameter specifying the kernel size M=2*L+1
            var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
            normalize (bool): Normalize kernel (Default value = True)

        Returns:
            kernel (np.ndarray): Kernel matrix of size M x M
        """
        taper = np.sqrt(1 / 2) / (L * var)
        axis = np.arange(-L, L + 1)
        gaussian1D = np.exp(-taper ** 2 * (axis ** 2))
        gaussian2D = np.outer(gaussian1D, gaussian1D)
        kernel_box = np.outer(np.sign(axis), np.sign(axis))
        kernel = kernel_box * gaussian2D
        if normalize:
            kernel = kernel / np.sum(np.abs(kernel))
        return kernel

    def compute_novelty_ssm(self, S, kernel=None, L=10, var=0.5, exclude=False):
        """Compute novelty function from SSM [FMP, Section 4.4.1]

        Notebook: C4/C4S4_NoveltySegmentation.ipynb

        Args:
            S (np.ndarray): SSM
            kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
            L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
            var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
            exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

        Returns:
            nov (np.ndarray): Novelty function
        """
        if kernel is None:
            kernel = self.compute_kernel_checkerboard_gaussian(L=L, var=var)
        N = S.shape[0]
        M = 2 * L + 1
        nov = np.zeros(N)
        # np.pad does not work with numba/jit
        S_padded = np.pad(S, L, mode='constant')

        for n in range(N):
            # Does not work with numba/jit
            nov[n] = np.sum(S_padded[n:n + M, n:n + M] * kernel)
        if exclude:
            right = np.min([L, N])
            left = np.max([0, N - L])
            nov[0:right] = 0
            nov[left:N] = 0

        return nov

    def compute_sm_from_light_features(self, light_features, L=21, H=1, L_smooth=16, tempo_rel_set=np.array([1]),
                                 shift_set=np.array([0]), strategy='relative', scale=True, thresh=0.15,
                                 penalty=0.0, binarize=False):
        """Compute an SSM

        adaption of Notebook: C4/C4S2_SSM-Thresholding.ipynb

        Args:
            light_features (str): light features np array
            L (int): Length of smoothing filter (Default value = 21)
            H (int): Downsampling factor (Default value = 5)
            L_smooth (int): Length of filter (Default value = 16)
            tempo_rel_set (np.ndarray):  Set of relative tempo values (Default value = np.array([1]))
            shift_set (np.ndarray): Set of shift indices (Default value = np.array([0]))
            strategy (str): Thresholding strategy (see :func:`libfmp.c4.c4s2_ssm.compute_sm_ti`)
                (Default value = 'relative')
            scale (bool): If scale=True, then scaling of positive values to range [0,1] (Default value = True)
            thresh (float): Treshold (meaning depends on strategy) (Default value = 0.15)
            penalty (float): Set values below treshold to value specified (Default value = 0.0)
            binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)

        Returns:
            X (np.ndarray): Feature sequence
            Fs_feature (scalar): Feature rate
            S_thresh (np.ndarray): SSM
        """

        # Chroma Feature Sequence and SSM
        X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(np.transpose(light_features), 30, filt_len=L, down_sampling=H)
        #X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

        # Compute SSM
        S, I = compute_sm_ti_normed(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
        S_thresh = libfmp.c4.threshold_matrix(S, thresh=thresh, strategy=strategy,
                                              scale=scale, penalty=penalty, binarize=binarize)
        return X, Fs_feature, S_thresh

    def compute_sm_from_chroma_stft(self, chroma_stft, L=21, H=1, L_smooth=16, tempo_rel_set=np.array([1]),
                                    shift_set=np.array([0]), strategy='relative', scale=True, thresh=0.15,
                                    penalty=0.0, binarize=False, shift = 0):
        """Compute an SSM

        Adaption of FMP-Notebook: C4/C4S2_SSM-Thresholding.ipynb

        Args:
            waveform (str): waveform
            L (int): Length of smoothing filter (Default value = 21)
            H (int): Downsampling factor (Default value = 0)
            L_smooth (int): Length of filter (Default value = 16)
            tempo_rel_set (np.ndarray):  Set of relative tempo values (Default value = np.array([1]))
            shift_set (np.ndarray): Set of shift indices (Default value = np.array([0]))
            strategy (str): Thresholding strategy (see :func:`libfmp.c4.c4s2_ssm.compute_sm_ti`)
                (Default value = 'relative')
            scale (bool): If scale=True, then scaling of positive values to range [0,1] (Default value = True)
            thresh (float): Treshold (meaning depends on strategy) (Default value = 0.15)
            penalty (float): Set values below treshold to value specified (Default value = 0.0)
            binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)
            shift (int): shift of corresponding Light Features

        Returns:
            x (np.ndarray): Audio signal
            x_duration (float): Duration of audio signal (seconds)
            X (np.ndarray): Feature sequence
            Fs_feature (scalar): Feature rate
            S_thresh (np.ndarray): SSM
            I (np.ndarray): Index matrix
        """

        # Waveform
        Fs = self.audio_extraction_config['sampling_rate']
        x_duration = chroma_stft.shape[0]

        # Chroma Feature Sequence and SSM
        C = np.transpose(chroma_stft.copy())
        Fs_C = Fs / self.audio_extraction_config['hop_length']

        #applying shift value
        if shift >0:
            C = C[:,shift:]
        elif shift < 0:
            C = C[:,:shift]

        # Chroma Feature Sequence and SSM
        X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C, filt_len=L, down_sampling=H)
        X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

        # Compute SSM
        S, I = libfmp.c4.compute_sm_ti(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
        S_thresh = libfmp.c4.threshold_matrix(S, thresh=thresh, strategy=strategy,
                                              scale=scale, penalty=penalty, binarize=binarize)
        return x_duration, X, Fs_feature, S_thresh, I


def compute_sm_ti_normed(X, Y, L=1, tempo_rel_set=np.asarray([1]), shift_set=np.asarray([0]), direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X (np.ndarray): First feature sequence
        Y (np.ndarray): Second feature sequence
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        shift_set (np.ndarray): Set of shift indices (Default value = np.asarray([0]))
        direction (int): Direction of smoothing (0: forward; 1: backward; 2: both directions) (Default value = 2)

    Returns:
        S_TI (np.ndarray): Transposition-invariant SM
        I_TI (np.ndarray): Transposition index matrix
    """
    for shift in shift_set:
        Y_cyc = libfmp.c4.shift_cyc_matrix(Y, shift)

        S_cyc = np.zeros((X.shape[1], X.shape[1]))

        for i in range(X.shape[1]):
            S_cyc[i,:] = np.linalg.norm(Y_cyc-np.tile(X[:,i],(X.shape[1],1)).transpose(), 2, axis = 0)

        S_cyc /= math.sqrt(X.shape[1])

        S_cyc = np.ones(S_cyc.shape) - S_cyc

        assert np.min(S_cyc) >= 0, "light SSM does have values smaller then 0"

        if direction == 0:
            S_cyc = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=0)
        if direction == 1:
            S_cyc = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=1)
        if direction == 2:
            S_forward = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift == shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0], S_cyc.shape[1])) * shift
        else:
            # jit does not like the following lines
            # I_greater = np.greater(S_cyc, S_TI)
            # I_greater = (S_cyc > S_TI)
            I_TI[S_cyc > S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)

    return S_TI, I_TI