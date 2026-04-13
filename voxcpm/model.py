"""VoxCPM model loading and inference utilities.

This module provides the core model interface for VoxCPM, handling
model initialization, audio preprocessing, and speech recognition.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Default sample rate expected by the model
DEFAULT_SAMPLE_RATE = 16000

# Supported audio formats
SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


class VoxCPM:
    """VoxCPM speech recognition model wrapper.

    Provides a high-level interface for loading VoxCPM checkpoints
    and running automatic speech recognition (ASR) inference.

    Args:
        model_dir: Path to the model checkpoint directory.
        device: Torch device string (e.g. 'cpu', 'cuda', 'cuda:0').
            Defaults to CUDA if available, otherwise CPU.
        dtype: Torch dtype for inference. Defaults to float16 on CUDA,
            float32 on CPU.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        # Resolve device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Resolve dtype
        if dtype is None:
            self.dtype = (
                torch.float16
                if self.device.type == "cuda"
                else torch.float32
            )
        else:
            self.dtype = dtype

        self._model = None
        self._processor = None
        logger.info(
            "VoxCPM initialised | model_dir=%s device=%s dtype=%s",
            self.model_dir,
            self.device,
            self.dtype,
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> "VoxCPM":
        """Load model weights into memory.

        Returns:
            self, to allow chaining: ``model = VoxCPM(...).load()``
        """
        if self._model is not None:
            logger.debug("Model already loaded, skipping.")
            return self

        logger.info("Loading VoxCPM from %s …", self.model_dir)
        try:
            # Import here so the module can be imported without heavy deps
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # type: ignore

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir), trust_remote_code=True
            )
            self._model = (
                AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(self.model_dir),
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load VoxCPM model from {self.model_dir}: {exc}"
            ) from exc

        logger.info("Model loaded successfully.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: Optional[str] = None,
        max_new_tokens: int = 448,
    ) -> str:
        """Transcribe speech from an audio file or raw waveform array.

        Args:
            audio: Path to an audio file **or** a 1-D float32 numpy array
                containing raw PCM samples at *sample_rate* Hz.
            sample_rate: Sample rate of the raw waveform (ignored when
                *audio* is a file path — the file's native rate is used).
            language: BCP-47 language tag (e.g. ``'zh'``, ``'en'``).
                When *None* the model auto-detects the language.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Transcribed text string.
        """
        if self._model is None:
            self.load()

        waveform, sr = self._load_audio(audio, sample_rate)

        inputs = self._processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            device=self.device, dtype=self.dtype
        )

        generate_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if language is not None:
            generate_kwargs["language"] = language

        with torch.inference_mode():
            predicted_ids = self._model.generate(
                input_features, **generate_kwargs
            )

        transcription: str = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_audio(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """Return (waveform_float32, sample_rate) from various input types."""
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32), sample_rate

        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{audio_path.suffix}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )

        try:
            import soundfile as sf  # type: ignore

            waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except ImportError as exc:
            raise ImportError(
                "'soundfile' is required to load audio files. "
                "Install it with: pip install soundfile"
            ) from exc

        # Downmix to mono if necessary
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        return waveform, sr

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        loaded = self._model is not None
        return (
            f"VoxCPM(model_dir={str(self.model_dir)!r}, "
            f"device={str(self.device)!r}, loaded={loaded})"
        )
