import numpy as np
import random
import pandas as pd
import torch.utils.data as data
import typing as tp
import soundfile as sf
import librosa
import cv2
from scipy import signal


PERIOD = 20  # 5#seconds
WINDOW_PERIOD = 5

BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


def mono_to_color(
    X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


"""
Expected shape [Batch , 39, 224, 180]

start =  s * (2500//self.PERIOD)//2
end   =  start + (2500//self.PERIOD)

"""


class SpectrogramDataset(data.Dataset):
    def __init__(
        self,
        file_list: tp.List[tp.List[str]],
        augmentation=False,
        img_size=224,
        waveform_transforms=None,
        spectrogram_transforms=None,
        melspectrogram_parameters={},
    ):
        self.file_list = (
            file_list  # list of list: [file_path, ebird_code, background_code]
        )
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.augmentation = augmentation
        self.PERIOD = PERIOD
        self.WINDOW_PERIOD = WINDOW_PERIOD

    def __len__(self):
        return len(self.file_list)

    def freq_mask(self, spec, F=30, num_masks=1, replace_with_zero=False):
        cloned = spec.copy()
        num_mel_channels = cloned.shape[0]

        for i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                # print('same')
                return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if replace_with_zero:
                cloned[f_zero:mask_end] = 0
            else:
                cloned[f_zero:mask_end] = cloned.mean()

        return cloned

    def time_mask(self, spec, T=40, num_masks=1, replace_with_zero=False):
        cloned = spec.copy()
        len_spectro = cloned.shape[1]

        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if replace_with_zero:
                cloned[:, t_zero:mask_end] = 0
            else:
                cloned[:, t_zero:mask_end] = cloned.mean()
        return cloned

    def __getitem__(self, idx: int):
        wav_path, ebird_code, background_code = self.file_list[idx]

        y, sr = sf.read(wav_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = 0  # np.random.randint(effective_length - len_y)
                new_y[start : start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start : start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        img_stack = []
        melspec_base = librosa.feature.melspectrogram(
            y, sr=sr, **self.melspectrogram_parameters
        )
        # melspec = librosa.pcen(melspec, sr=32000, hop_length=self.melspectrogram_parameters['hop_length'])
        melspec_base = librosa.power_to_db(melspec_base).astype(np.float32)

        for s in range(39):  # here it is like 20 + (20-1)
            # fmax_val = 16000

            start = s * (2500 // self.PERIOD) // 2
            end = start + (2500 // self.PERIOD)

            # print('start ',start)
            # print('end ',end)

            melspec = melspec_base[:, start:end]
            # fmax_val = 16000

            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(
                image, (int(width * self.img_size / height), self.img_size)
            )
            # image = cv2.resize(image,(self.img_size, self.img_size))
            image = np.flipud(image)
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)

            img_stack.append(image)

        #         labels = np.zeros(len(BIRD_CODE), dtype="i")
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        if ebird_code != "nocall":
            labels[BIRD_CODE[ebird_code]] = 1

            """## add the bg bird codes
            if not background_code != background_code:
                bg_val = background_code.split(',')
                for bg in bg_val:
                    if len(bg) > 0:
                        labels[BIRD_CODE[bg]] = .65"""

        return np.array(img_stack), labels


class TrainDataset(data.Dataset):
    def __init__(
        self,
        file_list: tp.List[tp.List[str]],
        train_df,
        bg_file_df,
        augmentation=True,
        img_size=224,
        waveform_transforms=None,
        spectrogram_transforms=None,
        melspectrogram_parameters={},
    ):
        self.file_list = (
            file_list  # list of list: [file_path, ebird_code, background_code]
        )
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.augmentation = augmentation

        self.train_df = train_df
        self.bird_species_list = (
            self.train_df["ebird_code"].value_counts().index.to_list()
        )
        self.PERIOD = PERIOD
        self.North_Am_bg = (
            bg_file_df  # self.train_df[train_df['ebird_code'] == 'nocall']
        )

    def __len__(self):
        return len(self.file_list)

    def freq_mask(self, spec, F=30, num_masks=1, replace_with_zero=False):
        cloned = spec.copy()
        num_mel_channels = cloned.shape[0]

        for i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                # print('same')
                return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if replace_with_zero:
                cloned[f_zero:mask_end] = 0
            else:
                cloned[f_zero:mask_end] = cloned.mean()

        return cloned

    def random_wav_selector(self, y, sr, PERIOD):
        len_y = len(y)
        effective_length = int(sr * PERIOD)
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start : start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start : start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)
        return y

    def random_wav_selector_noise(self, y, sr, PERIOD):
        len_y = len(y)
        effective_length = int(sr * PERIOD)
        if len_y < effective_length:

            ###get the count needed to be added
            counts = (effective_length // len_y) + 1
            final_y = y
            for i in range(counts):
                final_y = np.concatenate([final_y, y])
            y = final_y[:effective_length]
            y = y.astype(np.float32)

        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start : start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)
        return y

    def time_mask(self, spec, T=40, num_masks=1, replace_with_zero=False):
        cloned = spec.copy()
        len_spectro = cloned.shape[1]

        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if replace_with_zero:
                cloned[:, t_zero:mask_end] = 0
            else:
                cloned[:, t_zero:mask_end] = cloned.mean()
        return cloned

    def __getitem__(self, idx: int):
        wav_path, ebird_code, background_code = self.file_list[idx]

        y, sr = sf.read(wav_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            y = self.random_wav_selector(y, sr, self.PERIOD)

        # beginning of augmentation
        no_of_birds_aug = random.choice([0, 0, 1, 2, 3, 4])

        y_birds = []
        selected_birds = ["nocall", ebird_code]
        added_bg = []

        # background_code
        if no_of_birds_aug != 0:

            for bd in range(no_of_birds_aug):
                while True:
                    rand_bird = random.choice(self.bird_species_list)
                    if rand_bird not in selected_birds:
                        selected_birds.append(rand_bird)
                        break
                ##select a file based on the rand_bird choice
                df_t = self.train_df[self.train_df["ebird_code"] == rand_bird]
                sam = df_t.sample()
                added_bg.append(sam["background_code"].values)
                y_r, _ = sf.read(
                    sam["file_path"].values[0]
                )  # sf.read(TRAIN_RESAMPLED_AUDIO_DIRS / rand_bird / str(sam['xc_id'].values[0])+'.wav')

                y_birds.append(self.random_wav_selector(y_r, sr, self.PERIOD))
            sam = self.North_Am_bg.sample()
            fpp_noise = sam["file_path"].values[0]
            y_r_noise, _ = sf.read(fpp_noise)
            y_birds.append(self.random_wav_selector_noise(y_r_noise, sr, self.PERIOD))
            vv = [random.random() for ii in range(len(y_birds) + 1)]
            vv_l = [
                ii / sum(vv) for ii in vv
            ]  # this is the random number that need to be multiplid
            y_final = y * vv_l[0]

            for count, ii in enumerate(y_birds):
                y_final += ii * vv_l[count + 1]
        else:
            ## sam species augmented
            y_birds = []
            selected_birds = ["nocall", ebird_code]
            df_t = self.train_df[self.train_df["ebird_code"] == ebird_code]
            sam = df_t.sample()
            y_r, _ = sf.read(sam["file_path"].values[0])
            added_bg.append(sam["background_code"].values)
            y_birds.append(self.random_wav_selector(y_r, sr, self.PERIOD))
            sam = self.North_Am_bg.sample()
            fpp_noise = sam["file_path"].values[0]
            y_r_noise, _ = sf.read(fpp_noise)
            y_birds.append(self.random_wav_selector_noise(y_r_noise, sr, self.PERIOD))

            vv = [random.random() for ii in range(len(y_birds) + 1)]
            vv_l = [
                ii / sum(vv) for ii in vv
            ]  # this is the random number that need to be multiplid

            # print('len y_birds ',len(y_birds))
            # print('len vvs ',len(vv_l))
            y_final = y * vv_l[0]
            for count, ii in enumerate(y_birds):
                y_final += ii * vv_l[count + 1]

        cutOff = random.randrange(4000, 16000, 1000)
        # print('lowpass ',cutOff)
        # print(vv_l)
        nyq = 0.5 * sr
        N = 6  # Filter order
        fc = cutOff / nyq  # Cutoff frequency normal
        b, a = signal.butter(N, fc)

        y_final = signal.filtfilt(b, a, y_final)

        img_stack = []
        # print('len of y ',y_final.shape)
        melspec_base = librosa.feature.melspectrogram(
            y_final, sr=sr, **self.melspectrogram_parameters
        )
        # melspec = librosa.pcen(melspec, sr=32000, hop_length=self.melspectrogram_parameters['hop_length'])
        melspec_base = librosa.power_to_db(melspec_base).astype(np.float32)

        for s in range(39):  # here it is like 20 + (20-1)
            # fmax_val = 16000

            start = s * (2500 // self.PERIOD) // 2
            end = start + (2500 // self.PERIOD)

            # print('start ',start)
            # print('end ',end)

            melspec = melspec_base[:, start:end]
            # print('melspec shape ',s * (2500//4))
            # print('melspec shape ',s+1 * (2500//4))
            # print('melspec shape ',melspec.shape)

            if (
                self.augmentation
            ):  # https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
                n = random.random()
                # print(melspec.shape)
                if n <= 0.2:
                    # only frequency
                    # print('freq')
                    melspec = self.freq_mask(
                        melspec, F=20, num_masks=2, replace_with_zero=False
                    )
                elif n <= 0.4:
                    # only time
                    # print('time')
                    melspec = self.time_mask(
                        melspec, T=20, num_masks=3, replace_with_zero=False
                    )
                elif n <= 0.6:
                    # print('comb')
                    # both time and frequency
                    melspec = self.time_mask(
                        melspec, T=20, num_masks=2, replace_with_zero=False
                    )
                    melspec = self.freq_mask(
                        melspec, F=20, num_masks=2, replace_with_zero=False
                    )

            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(
                image, (int(width * self.img_size / height), self.img_size)
            )
            # image = cv2.resize(image,(self.img_size, self.img_size))
            image = np.flipud(image)
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)
            img_stack.append(image)
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        if ebird_code != "nocall":
            labels[BIRD_CODE[ebird_code]] = 1

        for ii in selected_birds:
            if ii == "nocall":
                continue
            labels[BIRD_CODE[ii]] = 1
        return np.array(img_stack), labels  # np.flipud(image)# image
