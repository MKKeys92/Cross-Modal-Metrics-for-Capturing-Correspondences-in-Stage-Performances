import numpy as np
import Sources.Preprocessing.Light.AbstractionLayers as AL
from types import SimpleNamespace

class DataGenerator:
    def __init__(self, runtime_args, lighting_abstraction_layer):
        self.args_gen = SimpleNamespace()
        self.reset_all_generators()
        for k,v in runtime_args.items():
            if k in ['base_gen', 'pan_gen', 'tilt_gen', 'hue_gen', 'sat_gen', 'dim_gen']:
                self.args_gen.__dict__[k] = v

        self.lighting_abstraction_layer = lighting_abstraction_layer

    def reset_all_generators(self):
        self.set_base_gen('')
        self.set_pan_gen('')
        self.set_tilt_gen('')
        self.set_hue_gen('')
        self.set_sat_gen('')
        self.set_dim_gen('')

    def onbeat_sawtooth(self,beats):
        curve = beats.copy().flatten()
        beats_idx = np.argwhere(curve == 1)
        last_b_idx = 0
        l = beats.shape[0]
        for b_idx in beats_idx:
            b_idx = b_idx[0]
            d = b_idx - last_b_idx
            if (last_b_idx) == 0:
                curve[0:b_idx + 1] = np.linspace(0, 1, num=d + 1)
            else:
                d_2 = int(d / 2)
                d_half = last_b_idx + d_2
                curve[last_b_idx:d_half + 1] = np.linspace(1, 0, num=d_2 + 1)
                curve[d_half: b_idx + 1] = np.linspace(0, 1, num=d - d_2 + 1)
            last_b_idx = b_idx
        if last_b_idx != l:
            curve[last_b_idx:] = np.linspace(1, 0, num=l - last_b_idx)

        return curve

    def offbeat_sawtooth(self,beats):
        curve = self.onbeat_sawtooth(beats)
        ones = np.ones(curve.shape)
        curve = ones - curve
        return curve

    def get_onset_curve(self,onset):
        curve = onset.copy().flatten()
        for i in range(1,len(curve)):
            curve[i] = curve[i] + curve[i-1]
        curve /= curve[-1]
        return curve

    def set_base_gen(self, val):
        self.args_gen.base_gen = val

    def set_dim_gen(self,val):
        self.args_gen.dim_gen = val

    def set_hue_gen(self,val):
        self.args_gen.hue_gen = val

    def set_sat_gen(self,val):
        self.args_gen.sat_gen = val

    def set_pan_gen(self,val):
        self.args_gen.pan_gen = val

    def set_tilt_gen(self,val):
        self.args_gen.tilt_gen = val

    def generate_or_modify_data(self, r_in, audio_features):
        light = r_in

        if self.args_gen.base_gen == "random":
            light = np.random.rand(r_in.size)
            light = light.reshape(r_in.shape)
        elif self.args_gen.base_gen == "onset":
            curve = self.get_onset_curve(audio_features['onset_env'])
            light = np.tile(curve, (r_in.shape[1], 1)).transpose()[:r_in.shape[0],:]
        elif self.args_gen.base_gen == "chroma":
            light = np.zeros(r_in.shape)
            C = audio_features['chroma_stft']
            cr = np.tile(C,(1, int(np.floor(r_in.shape[1] / C.shape[1]))))[:r_in.shape[0],:]
            light[:,:cr.shape[1]] = cr
        elif self.args_gen.base_gen == "zeros":
            light = np.zeros(r_in.shape)
        elif self.args_gen.base_gen == "static":
            light = np.ones(r_in.shape)
        elif self.args_gen.base_gen == "rms":
            rms = audio_features['rms']
            rms = rms.flatten()
            normed_rms = rms / np.max(rms)
            light = np.tile(normed_rms, (r_in.shape[1], 1)).transpose()[:r_in.shape[0],:]
        elif self.args_gen.base_gen == "onbeat":
            beats = audio_features['onset_beat']
            ob_saw = self.onbeat_sawtooth(beats)
            light = np.tile(ob_saw, (r_in.shape[1], 1)).transpose()[:r_in.shape[0],:]
        elif self.args_gen.base_gen == "offbeat":
            beats = audio_features['onset_beat']
            ob_saw = self.offbeat_sawtooth(beats)
            light = np.tile(ob_saw, (r_in.shape[1], 1)).transpose()[:r_in.shape[0],:]
        elif self.args_gen.base_gen != '':
            raise NotImplementedError()

        al = AL.get_abstraction_layer(self.lighting_abstraction_layer)
        b_s, hue_s, sat_s, pan_s, tilt_s = al.get_attribute_sizes()

        # dimmer variants
        if self.args_gen.dim_gen == 'onbeat':
            beats = audio_features['onset_beat']
            ob_saw = self.onbeat_sawtooth(beats)
            ob_saw = np.tile(ob_saw, (b_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_brightness(light,ob_saw)
        elif self.args_gen.dim_gen == 'offbeat':
            beats = audio_features['onset_beat']
            ob_saw = self.offbeat_sawtooth(beats)
            ob_saw = np.tile(ob_saw, (b_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_brightness(light, ob_saw)
        elif self.args_gen.dim_gen == 'rms':
            rms = audio_features['rms']
            rms = rms.flatten()
            normed_rms = rms / np.max(rms)
            normed_rms = np.tile(normed_rms, (b_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_brightness(light, normed_rms)
        elif self.args_gen.dim_gen == 'random':
            r = np.random.rand(r_in.shape[0], b_s)
            light = al.set_brightness(light, r)
        elif self.args_gen.dim_gen == 'onset':
            onset = audio_features['onset_env']
            curve = self.get_onset_curve(onset)
            curve = np.tile(curve, (b_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_brightness(light, curve)
        elif self.args_gen.dim_gen != '':
            raise NotImplementedError()

        # position variants
        if self.args_gen.pan_gen == 'onset':
            onset = audio_features['onset_env']
            curve = self.get_onset_curve(onset)
            curve_p = np.tile(curve, (pan_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_pan(light, curve_p)
        elif self.args_gen.pan_gen == 'random':
            curve_p = np.random.rand(r_in.shape[0], pan_s)
            light = al.set_pan(light, curve_p)
        elif self.args_gen.pan_gen != '':
            raise NotImplementedError()

        if self.args_gen.tilt_gen == 'onset':
            onset = audio_features['onset_env']
            curve = self.get_onset_curve(onset)
            curve_t = np.tile(curve, (tilt_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_tilt(light, curve_t)
        elif self.args_gen.tilt_gen == 'random':
            curve_t = np.random.rand(r_in.shape[0], tilt_s)
            light = al.set_tilt(light, curve_t)
        elif self.args_gen.tilt_gen != '':
            raise NotImplementedError()

        # color variants
        if self.args_gen.hue_gen == 'onset':
            onset = audio_features['onset_env']
            curve = self.get_onset_curve(onset)
            curve_h = np.tile(curve, (hue_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_hue(light, curve_h)
        elif self.args_gen.hue_gen == 'random':
            curve_h = np.random.rand(r_in.shape[0], hue_s)
            light = al.set_hue(light, curve_h)
        elif self.args_gen.hue_gen != '':
            raise NotImplementedError()

        if self.args_gen.sat_gen == 'onset':
            onset = audio_features['onset_env']
            curve = self.get_onset_curve(onset)
            curve_s = np.tile(curve, (sat_s, 1)).transpose()[:r_in.shape[0],:]
            light = al.set_sat(light, curve_s)
        elif self.args_gen.sat_gen == 'random':
            curve_s = np.random.rand(r_in.shape[0], sat_s)
            light = al.set_sat(light, curve_s)
        elif self.args_gen.sat_gen != '':
            raise NotImplementedError()

        return light
