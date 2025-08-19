# Vector generation utilities

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