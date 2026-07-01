"""Embedding model registry (spec: Deep Embeddings).

Every backend implements EmbeddingModel and registers itself; the worker
just iterates over whatever is available. Missing deps degrade gracefully:
the model reports unavailable and is skipped (logged), so the pipeline
never hard-fails because one of eight model stacks isn't installed.

GPU: torch backends pick cuda/mps automatically; essentia-tensorflow uses
GPU if the TF build sees one.

Models:
  general        openl3, clap (LAION), music2vec, musicfm
  genre          discogs-effnet, musicnn
  self-supervised byola, wav2vec2 (music-adapted)
"""

import abc
import logging

import numpy as np

log = logging.getLogger("audiolens.embedder.registry")

REGISTRY: dict[str, type["EmbeddingModel"]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def _torch_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingModel(abc.ABC):
    name: str
    version: str
    dim: int

    _instance = None

    @classmethod
    def get(cls):
        """Lazy singleton — model weights load once per worker process."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def available(cls) -> bool:
        try:
            cls._check_deps()
            return True
        except ImportError as e:
            log.info("%s unavailable: %s", cls.name, e)
            return False

    @staticmethod
    def _check_deps() -> None: ...

    @abc.abstractmethod
    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Returns a single L2-normalizable vector of shape (dim,)."""


def _mean_pool(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0).astype(np.float32)


@register
class OpenL3Model(EmbeddingModel):
    name, version, dim = "openl3", "openl3-music-mel256-512", 512

    @staticmethod
    def _check_deps():
        import openl3  # noqa: F401

    def __init__(self):
        import openl3
        self._m = openl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="music", embedding_size=self.dim
        )

    def embed(self, y, sr):
        import openl3
        emb, _ts = openl3.get_audio_embedding(
            y, sr, model=self._m, hop_size=1.0, verbose=False
        )
        return _mean_pool(emb)


@register
class CLAPModel(EmbeddingModel):
    name, version, dim = "clap", "laion-clap-music_audioset", 512

    @staticmethod
    def _check_deps():
        import laion_clap  # noqa: F401

    def __init__(self):
        import laion_clap
        self._m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self._m.load_ckpt(model_id=1)  # music_audioset checkpoint
        self._m.to(_torch_device())

    def embed(self, y, sr):
        import librosa
        if sr != 48000:
            y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        return self._m.get_audio_embedding_from_data(
            x=y[None, :], use_tensor=False
        )[0].astype(np.float32)


@register
class Music2VecModel(EmbeddingModel):
    name, version, dim = "music2vec", "m-a-p/music2vec-v1", 768

    @staticmethod
    def _check_deps():
        import torch  # noqa: F401
        import transformers  # noqa: F401

    def __init__(self):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.device = _torch_device()
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/music2vec-v1")
        self._m = AutoModel.from_pretrained("m-a-p/music2vec-v1").to(self.device).eval()
        self._torch = torch

    def embed(self, y, sr):
        import librosa
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
        with self._torch.no_grad():
            inp = self._fe(y, sampling_rate=24000, return_tensors="pt").to(self.device)
            out = self._m(**inp, output_hidden_states=True)
            # average over layers + time (per model card)
            h = self._torch.stack(out.hidden_states).mean(dim=0).mean(dim=1)
        return h[0].cpu().numpy().astype(np.float32)


@register
class MusicFMModel(EmbeddingModel):
    name, version, dim = "musicfm", "minzwon/musicfm-msd", 750

    @staticmethod
    def _check_deps():
        import musicfm  # noqa: F401  (pip install from github minzwon/musicfm)

    def __init__(self):
        import torch
        from musicfm.model.musicfm_25hz import MusicFM25Hz
        self.device = _torch_device()
        self._m = MusicFM25Hz().to(self.device).eval()
        self._torch = torch

    def embed(self, y, sr):
        import librosa
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
        with self._torch.no_grad():
            x = self._torch.from_numpy(y[None, :]).float().to(self.device)
            _, hidden = self._m.get_predictions(x)
        return hidden.mean(dim=1)[0].cpu().numpy().astype(np.float32)


class _EssentiaTFModel(EmbeddingModel):
    """Base for essentia-tensorflow .pb graph models (downloaded by make models)."""
    graph_file: str
    output_node: str
    target_sr = 16000

    @staticmethod
    def _check_deps():
        import essentia.standard  # noqa: F401

    def __init__(self):
        import os
        from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN
        path = os.path.join(os.environ.get("MODEL_DIR", "/models"), self.graph_file)
        if "effnet" in self.graph_file:
            self._m = TensorflowPredictEffnetDiscogs(graphFilename=path, output=self.output_node)
        else:
            self._m = TensorflowPredictMusiCNN(graphFilename=path, output=self.output_node)

    def embed(self, y, sr):
        import librosa
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        return _mean_pool(np.array(self._m(y.astype(np.float32))))


@register
class DiscogsEffnetModel(_EssentiaTFModel):
    name, version, dim = "discogs-effnet", "discogs-effnet-bs64-1", 1280
    graph_file = "discogs-effnet-bs64-1.pb"
    output_node = "PartitionedCall:1"  # penultimate = embedding


@register
class MusiCNNModel(_EssentiaTFModel):
    name, version, dim = "musicnn", "msd-musicnn-1", 200
    graph_file = "msd-musicnn-1.pb"
    output_node = "model/dense/BiasAdd"


@register
class BYOLAModel(EmbeddingModel):
    name, version, dim = "byola", "byol-a-v2-2048", 2048

    @staticmethod
    def _check_deps():
        import byol_a2  # noqa: F401  (pip install from github nttcslab/byol-a)

    def __init__(self):
        import torch
        from byol_a2.augmentations import PrecomputedNorm
        from byol_a2.common import load_yaml_config
        from byol_a2.models import AudioNTT2022, load_pretrained_weights
        import os
        cfg = load_yaml_config(os.path.join(os.environ.get("MODEL_DIR", "/models"), "byola/config_v2.yaml"))
        self.device = _torch_device()
        self._m = AudioNTT2022(d=cfg.feature_d).to(self.device).eval()
        load_pretrained_weights(self._m, os.path.join(os.environ.get("MODEL_DIR", "/models"), "byola/AudioNTT2022-BYOLA-64x96d2048.pth"))
        self._norm = PrecomputedNorm([-5.4919, 5.0389])
        self._cfg = cfg
        self._torch = torch

    def embed(self, y, sr):
        import librosa
        import torchaudio
        cfg = self._cfg
        if sr != cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate)
        to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max,
        )
        with self._torch.no_grad():
            lms = self._norm((to_mel(self._torch.from_numpy(y).float()) + self._torch.finfo().eps).log())
            return self._m(lms.unsqueeze(0).unsqueeze(0).to(self.device))[0].cpu().numpy().astype(np.float32)


@register
class Wav2Vec2Model(EmbeddingModel):
    name, version, dim = "wav2vec2", "facebook/wav2vec2-base-960h+music-adapt", 768

    @staticmethod
    def _check_deps():
        import torch  # noqa: F401
        import transformers  # noqa: F401

    def __init__(self):
        import os
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        ckpt = os.environ.get("WAV2VEC2_CKPT", "facebook/wav2vec2-base-960h")
        self.device = _torch_device()
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        self._m = AutoModel.from_pretrained(ckpt).to(self.device).eval()
        self._torch = torch

    def embed(self, y, sr):
        import librosa
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        with self._torch.no_grad():
            inp = self._fe(y, sampling_rate=16000, return_tensors="pt").to(self.device)
            out = self._m(**inp)
        return out.last_hidden_state.mean(dim=1)[0].cpu().numpy().astype(np.float32)


def embed_all(y: np.ndarray, sr: int, models: list[str] | None = None) -> dict[str, dict]:
    """Run every (requested+available) model. Returns {name: {vector, dim, version}}."""
    out = {}
    for name, cls in REGISTRY.items():
        if models and name not in models:
            continue
        if not cls.available():
            continue
        try:
            v = cls.get().embed(y, sr)
            out[name] = {"vector": v.tolist(), "dim": int(v.shape[0]), "version": cls.version}
        except Exception as e:  # noqa: BLE001
            log.exception("embedding %s failed: %s", name, e)
    return out
