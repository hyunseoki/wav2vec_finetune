import torch
import librosa
import os


def read_audio(mp3_path, target_sr=16000):
    """
    Loads an mp3 audio file and resamples it to 16kHz 
    Required for needed for Wav2Vec2 training
    """
    audio, sr = librosa.load(mp3_path, sr=32000)
    audio_array = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_array


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir, df, processor, is_test=False):
        self.wav_dir = wav_dir
        self.df = df
        self.processor = processor
        self.is_test = is_test


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        wav_fn = os.path.join(self.wav_dir, self.df.loc[idx]['file_name'])
        assert os.path.isfile(wav_fn), wav_fn
        audio = read_audio(wav_fn)
        audio = self.processor(audio, sampling_rate=16000).input_values[0]
        # Return -1 for label if in test-only mode
        if self.is_test:
            return {'audio': audio, 'label': -1}
        else:
            # If we are training/validating, also process the labels (actual sentences)
            with self.processor.as_target_processor():
                labels = self.processor(self.df.loc[idx]['transcription']).input_ids
            return {'audio': audio, 'label': labels}