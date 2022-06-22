import numpy as np
import pandas as pd


class OkrProcessor:
    def __init__(self, okr_fn, exclude_outlier_thre=None):
        self.okr_fn = okr_fn
        self.okr_file = pd.read_csv(self.okr_fn, delimiter='\t', header=None)
        self.okr_time = self.okr_file[0]
        self.le_pos = self.okr_file[4]
        self.re_pos = self.okr_file[5]
        if exclude_outlier_thre:
            self.le_pos[self.le_pos <= exclude_outlier_thre] = np.nan
            self.re_pos[self.re_pos <= exclude_outlier_thre] = np.nan
        okr_time_diff = np.maximum(self.okr_time.diff(), 0)
        okr_time_diff[0] = self.okr_time[0]
        self.okr_time_nonNeg = np.cumsum(okr_time_diff)
        self.le_pos.index = self.okr_time_nonNeg
        self.re_pos.index = self.okr_time_nonNeg
        self.phase_idx_array = self.okr_file[24]
        self.phase_idx_array.index = self.okr_time_nonNeg
        self.phase_idx = np.unique(self.phase_idx_array)
        self.phase_idx = self.phase_idx[self.phase_idx!=0]

    def detect_saccade(self, slope_thre, movWinSize):
        le_diff2 = self.le_pos.diff().diff().abs()
        le_saccade_raw = pd.Series(np.roll(le_diff2 > slope_thre, -1), dtype=int).rolling(movWinSize).mean() > 0
        le_saccade_raw = pd.Series(le_saccade_raw, dtype=int)
        le_saccade_st = np.where(le_saccade_raw.diff() == 1)[0]
        le_saccade_ed = np.where(le_saccade_raw.diff() == -1)[0]
        self.le_saccade = le_saccade_raw * 0
        self.le_saccade[le_saccade_st] = np.sign(
            self.le_pos.iloc[le_saccade_st].to_numpy() - self.le_pos.iloc[le_saccade_ed].to_numpy())
        self.le_slowphase = self.le_pos.diff()
        self.le_slowphase.iloc[le_saccade_raw > 0] = 0
        self.le_slowphase = self.le_slowphase.cumsum().to_numpy()

        re_diff2 = self.re_pos.diff().diff().abs()
        re_saccade_raw = pd.Series(np.roll(re_diff2 > slope_thre, -1), dtype=int).rolling(movWinSize).mean() > 0
        re_saccade_raw = pd.Series(re_saccade_raw, dtype=int)
        re_saccade_st = np.where(re_saccade_raw.diff() == 1)[0]
        re_saccade_ed = np.where(re_saccade_raw.diff() == -1)[0]
        self.re_saccade = re_saccade_raw * 0
        self.re_saccade[re_saccade_st] = np.sign(
            self.re_pos.iloc[re_saccade_st].to_numpy() - self.re_pos.iloc[re_saccade_ed].to_numpy())
        self.re_slowphase = self.re_pos.diff()
        self.re_slowphase.iloc[re_saccade_raw > 0] = 0
        self.re_slowphase = self.re_slowphase.cumsum().to_numpy()
        processed = {}
        processed['le_saccade'] = self.le_saccade
        processed['re_saccade'] = self.re_saccade
        processed['le_slowphase'] = self.le_slowphase
        processed['re_slowphase'] = self.re_slowphase
        return processed

    def compute_okr_statistics(self, slope_thre=2, movWinSize=5, processed_data={}):
        if processed_data:
             self.le_saccade = processed_data['le_saccade']
             self.re_saccade = processed_data['re_saccade']
             self.le_slowphase = processed_data['le_slowphase']
             self.re_slowphase = processed_data['re_slowphase']
        else:
            self.detect_saccade(slope_thre, movWinSize)
        okr_statistic = pd.DataFrame(
            columns=['phase', 'repeat', 'le_okr_speed', 'le_pos_saccade_num', 'le_neg_saccade_num',
                     're_okr_speed', 're_pos_saccade_num', 're_neg_saccade_num', ])
        counter = 0
        for i in self.phase_idx:
            phase_diff = np.diff(np.array(self.phase_idx_array == i, dtype=int))
            st_pos = np.where(phase_diff == 1)[0] + 1
            ed_pos = np.where(phase_diff == -1)[0] + 1
            if len(ed_pos) == 0:
                ed_pos = np.array([len(phase_diff)])
            if st_pos[0]>ed_pos[0]:
                st_pos_buffer = st_pos
                st_pos = ed_pos*1
                ed_pos = st_pos_buffer*1
            repeat_i = 0
            for j, k in zip(st_pos, ed_pos):
                iter_repeat_statistic = {}
                le_seg = self.le_slowphase[j:k]
                le_seg = np.array(le_seg[~np.isnan(le_seg)])
                le_slope = (le_seg[-1] - le_seg[0]) / (self.okr_time_nonNeg[k] - self.okr_time_nonNeg[j])
                le_pos_saccade_num = np.sum(self.le_saccade[j:k] == 1)
                le_neg_saccade_num = np.sum(self.le_saccade[j:k] == -1)
                iter_repeat_statistic['le_okr_speed'] = le_slope
                iter_repeat_statistic['le_pos_saccade_num'] = le_pos_saccade_num
                iter_repeat_statistic['le_neg_saccade_num'] = le_neg_saccade_num

                re_seg = self.re_slowphase[j:k]
                re_seg = np.array(re_seg[~np.isnan(re_seg)])
                re_slope = (re_seg[-1] - re_seg[0]) / (self.okr_time_nonNeg[k] - self.okr_time_nonNeg[j])
                re_pos_saccade_num = np.sum(self.re_saccade[j:k] == 1)
                re_neg_saccade_num = np.sum(self.re_saccade[j:k] == -1)
                iter_repeat_statistic['re_okr_speed'] = re_slope
                iter_repeat_statistic['re_pos_saccade_num'] = re_pos_saccade_num
                iter_repeat_statistic['re_neg_saccade_num'] = re_neg_saccade_num

                iter_repeat_statistic['phase'] = i
                iter_repeat_statistic['repeat'] = repeat_i
                okr_statistic = pd.concat([okr_statistic, pd.DataFrame([iter_repeat_statistic, ])])
                repeat_i += 1
                counter += 1
        return okr_statistic
