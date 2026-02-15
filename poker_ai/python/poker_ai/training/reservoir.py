"""Reservoir sampling buffer for Average Strategy (supervised) training.

Uses pre-allocated numpy arrays for zero-allocation batch operations.
"""

import io
import threading

import numpy as np
from dataclasses import dataclass


@dataclass
class SLTransition:
    """Single SL transition (kept for API compatibility)."""
    obs: np.ndarray
    action_history: np.ndarray
    history_length: int
    action: int
    legal_mask: np.ndarray


class ReservoirBuffer:
    """Reservoir sampling buffer using pre-allocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int = 462, max_seq_len: int = 30, num_actions: int = 9, history_dim: int = 11):
        self.capacity = capacity
        self.size = 0
        self.total_seen = 0
        self.rng = np.random.default_rng()
        self._lock = threading.Lock()

        # Pre-allocated arrays
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_history = np.zeros((capacity, max_seq_len, history_dim), dtype=np.float32)
        self.history_length = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.legal_mask = np.zeros((capacity, num_actions), dtype=bool)

    def push(self, transition: SLTransition):
        self.total_seen += 1
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            idx = int(self.rng.integers(0, self.total_seen))
            if idx >= self.capacity:
                return
            i = idx

        self.obs[i] = transition.obs
        self.action_history[i] = transition.action_history
        self.history_length[i] = transition.history_length
        self.actions[i] = transition.action
        self.legal_mask[i] = transition.legal_mask

    def push_batch(
        self,
        obs: np.ndarray,
        action_history: np.ndarray,
        history_length: np.ndarray,
        actions: np.ndarray,
        legal_mask: np.ndarray,
    ):
        """Push a batch using reservoir sampling."""
        n = len(obs)
        if n == 0:
            return

        with self._lock:
            if self.capacity <= 0:
                self.total_seen += n
                return

            # Fill any remaining capacity with contiguous writes.
            fill = min(self.capacity - self.size, n)
            if fill > 0:
                dest = slice(self.size, self.size + fill)
                self.obs[dest] = obs[:fill]
                self.action_history[dest] = action_history[:fill]
                self.history_length[dest] = history_length[:fill]
                self.actions[dest] = actions[:fill]
                self.legal_mask[dest] = legal_mask[:fill]
                self.size += fill

            # Once full, perform vectorized reservoir replacement for the remainder.
            remaining = n - fill
            if remaining <= 0:
                self.total_seen += n
                return

            start_seen = self.total_seen + fill
            highs = np.arange(start_seen + 1, start_seen + remaining + 1, dtype=np.int64)
            replace_idx = self.rng.integers(0, highs, size=remaining)
            keep = replace_idx < self.capacity
            self.total_seen += n
            if not np.any(keep):
                return

            target = replace_idx[keep]
            source = np.flatnonzero(keep) + fill

            # Preserve sequential semantics when multiple updates hit the same slot:
            # the last seen sample for a slot should win.
            if target.size > 1:
                order = np.argsort(target, kind="stable")
                target_sorted = target[order]
                source_sorted = source[order]
                last = np.ones(target_sorted.shape[0], dtype=bool)
                last[:-1] = target_sorted[:-1] != target_sorted[1:]
                target = target_sorted[last]
                source = source_sorted[last]

            self.obs[target] = obs[source]
            self.action_history[target] = action_history[source]
            self.history_length[target] = history_length[source]
            self.actions[target] = actions[source]
            self.legal_mask[target] = legal_mask[source]

    def sample_arrays(self, batch_size: int) -> tuple:
        """Sample and return raw numpy arrays.

        Returns: (obs, ah, ah_len, actions, masks)
        """
        with self._lock:
            indices = np.random.randint(0, self.size, size=batch_size)
            return (
                self.obs[indices].copy(),
                self.action_history[indices].copy(),
                self.history_length[indices].copy(),
                self.actions[indices].copy(),
                self.legal_mask[indices].copy(),
            )

    def sample(self, batch_size: int) -> list[SLTransition]:
        """Sample transitions (legacy API)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return [
            SLTransition(
                obs=self.obs[i],
                action_history=self.action_history[i],
                history_length=int(self.history_length[i]),
                action=int(self.actions[i]),
                legal_mask=self.legal_mask[i],
            )
            for i in indices
        ]

    def __len__(self) -> int:
        with self._lock:
            return self.size

    def save(self, path: str) -> None:
        """Save buffer contents to an .npz file (float16 for large arrays).

        Streams data in chunks to keep peak memory overhead under ~100MB.
        """
        import tempfile, os, zipfile
        with self._lock:
            n = self.size
            if n == 0:
                return
            total_seen = self.total_seen

        CHUNK = 50_000

        def _write_array(zf, name, arr, dtype=None):
            out_dtype = np.dtype(dtype) if dtype else arr.dtype
            with zf.open(f"{name}.npy", "w", force_zip64=True) as f:
                header_buf = io.BytesIO()
                np.lib.format.write_array_header_2_0(
                    header_buf,
                    {"descr": np.lib.format.dtype_to_descr(out_dtype),
                     "fortran_order": False,
                     "shape": arr.shape}
                )
                f.write(header_buf.getvalue())
                for start in range(0, len(arr), CHUNK):
                    end = min(start + CHUNK, len(arr))
                    chunk = arr[start:end]
                    if dtype and np.dtype(dtype) != arr.dtype:
                        chunk = chunk.astype(dtype)
                    f.write(chunk.tobytes())

        def _write_small(zf, name, val):
            buf = io.BytesIO()
            np.save(buf, np.array(val))
            zf.writestr(f"{name}.npy", buf.getvalue())

        dir_name = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".npz")
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_path, "w", allowZip64=True) as zf:
                _write_array(zf, "obs", self.obs[:n], np.float16)
                _write_array(zf, "action_history", self.action_history[:n], np.float16)
                _write_array(zf, "history_length", self.history_length[:n])
                _write_array(zf, "actions", self.actions[:n])
                _write_array(zf, "legal_mask", self.legal_mask[:n])
                _write_small(zf, "size", [n])
                _write_small(zf, "total_seen", [total_seen])
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self, path: str) -> None:
        """Load buffer contents from a .npz file."""
        data = np.load(path)
        n = int(data["size"][0])
        if n == 0:
            return
        n = min(n, self.capacity)
        with self._lock:
            self.obs[:n] = data["obs"][:n].astype(np.float32)
            self.action_history[:n] = data["action_history"][:n].astype(np.float32)
            self.history_length[:n] = data["history_length"][:n]
            self.actions[:n] = data["actions"][:n]
            self.legal_mask[:n] = data["legal_mask"][:n]
            self.size = n
            self.total_seen = int(data["total_seen"][0])
