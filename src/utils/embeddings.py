from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Singleton holders to avoid reloading on every call
_processor: Optional[Wav2Vec2Processor] = None 
_model: Optional[Wav2Vec2Model] = None

def _get_models() -> tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    global _processor, _model
    if _processor is None or _model is None:
        _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        _model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _model.eval()
    return _processor, _model

def get_audio_embedding(audio_path: Union[str, Path]) -> np.ndarray:
    """Generate a 768-dim embedding for an audio file using wav2vec2-base.

    Args:
        audio_path: Path to an audio file. Will be resampled to 16kHz mono.

    Returns:
        Numpy array of shape (768, ) representing the embedding.
    """
    processor, model = _get_models()
    audio, _ = librosa.load(str(audio_path), sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state.squeeze(0)
    emb = hidden.mean(dim=0).cpu().numpy()
    return emb

class AudioEmbedding:
    """Wrapper for the wav2vec2-base audio embedding."""
    def get_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
        return get_audio_embedding(audio_path)

# def OpenL3Embedding():
#     """Wrapper for OpenL3 audio embeddings."""
#     from openl3 import get_audio_embedding as openl3_get_embedding

#     class _OpenL3Embedding:
#         def get_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
#             audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
#             emb, _ = openl3_get_embedding(
#                 audio,
#                 sr,
#                 content_type="music",
#                 input_repr="mel256",
#                 embedding_size=512,
#                 center=True,
#                 hop_size=0.1,
#             )
#             return emb.mean(axis=0)
#     return _OpenL3Embedding()

def YAMNetEmbedding():
    """Wrapper for YAMNet audio embeddings."""
    import tensorflow as tf
    import tensorflow_hub as hub

    class _YAMNetEmbedding:
        def __init__(self):
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        def get_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            waveform = audio.reshape(-1)
            waveform = waveform.astype(np.float32)
            scores, embeddings, spectrogram = self.model(waveform)
            return np.mean(embeddings.numpy(), axis=0)
    return _YAMNetEmbedding()

def PaNNsEmbedding():
    """Wrapper for PaNNs audio embeddings."""
    from transformers import AutoProcessor, AutoModel

    class _PaNNsEmbedding:
        def __init__(self):
            self.processor = AutoProcessor.from_pretrained("m-a-p/PANNs_Cnn14_DecisionLevelMax")
            self.model = AutoModel.from_pretrained("m-a-p/PANNs_Cnn14_DecisionLevelMax")
            self.model.eval()

        def get_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
            audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
            inputs = self.processor(audio, sampling_rate=32000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            emb = outputs.embeddings.squeeze(0).cpu().numpy()
            return emb
    return _PaNNsEmbedding()

def PaSSTEmbedding():
    """Wrapper for PaSST audio embeddings."""
    from transformers import AutoProcessor, AutoModel

    class _PaSSTEmbedding:
        def __init__(self):
            self.processor = AutoProcessor.from_pretrained("m-a-p/PASST-slim")
            self.model = AutoModel.from_pretrained("m-a-p/PASST-slim")
            self.model.eval()

        def get_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
            audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
            inputs = self.processor(audio, sampling_rate=32000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0)
            emb = hidden.mean(dim=0).cpu().numpy()
            return emb
    return _PaSSTEmbedding()
