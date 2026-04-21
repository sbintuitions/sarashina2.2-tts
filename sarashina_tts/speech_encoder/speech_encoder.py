import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from typing import List, Union

from .campplus import CAMPPlus

class SpeechEncoder:
    """Speech Encoder.
    This class provides functionality to encode audio waveforms into embeddings.
    Currently using PyTorch implementation of the camplus model. 
    """
    
    def __init__(self,
                 camplus_model: str,
                 device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = CAMPPlus()
        state_dict = torch.load(
            camplus_model, 
            map_location="cpu", 
            weights_only=False
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def encode(self, speech_16k: torch.Tensor):
        """ Encode audio to embeddings.
        Args:
            speech_16k: Audio waveform tensor of shape (1, T) or (T,). Sampled at 16kHz.
        Returns:
            torch.Tensor: Embedding tensor of shape (1, 192)
        """
        if speech_16k.dim() == 1:
            speech_16k = speech_16k.unsqueeze(0)
            
        feat = kaldi.fbank(speech_16k,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat = feat.unsqueeze(0).to(self.device)
        
        embedding = self.model(feat)
        return embedding
    
    def encode_from_path(self, speech_path: str):
        """ Encode audio from file path to embeddings.
        
        Will load the audio file, resample it to 16kHz if necessary, and then encode it.
        Args:
            speech_path (str): Path to the audio file.
        Returns:
            torch.Tensor: Embedding tensor of shape (1, 192)
        """
        speech_np, sample_rate = sf.read(speech_path, always_2d=True)
        speech_np = speech_np[:, 0:1]  # force mono (keep 2D shape)
        speech = torch.from_numpy(speech_np.T).float()
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech)
        return self.encode(speech)

    @torch.no_grad()
    def encode_batch(self, speech_list: List[torch.Tensor]):
        """ Encode a batch of audio waveforms to embeddings.
        Args:
            speech_list: List of audio waveform tensors of shape (1, T) or (T,). Sampled at 16kHz.
        Returns:
            torch.Tensor: Embedding tensor of shape (B, 192)
        """
        features = []
        for speech in speech_list:
            if speech.dim() == 1:
                speech = speech.unsqueeze(0)
            feat = kaldi.fbank(speech,
                               num_mel_bins=80,
                               dither=0,
                               sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            features.append(feat)

        # Pad features for batch processing
        max_len = max(feat.shape[0] for feat in features)
        padded_features = torch.zeros(len(features), max_len, 80)
        for i, feat in enumerate(features):
            padded_features[i, :feat.shape[0], :] = feat
            
        padded_features = padded_features.to(self.device)
        embeddings = self.model(padded_features)
        return embeddings

    def encode_batch_from_paths(self, speech_paths: List[str]):
        """ Encode multiple audio files from paths to embeddings in a batch.
        Args:
            speech_paths (List[str]): List of paths to audio files.
        Returns:
            torch.Tensor: Embedding tensor of shape (B, 192)
        """
        speech_list = []
        for speech_path in speech_paths:
            speech_np, sample_rate = sf.read(speech_path, always_2d=True)
            speech_np = speech_np[:, 0:1]  # force mono (keep 2D shape)
            speech = torch.from_numpy(speech_np.T).float()
            if sample_rate != 16000:
                speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech)
            speech_list.append(speech)
        return self.encode_batch(speech_list)
