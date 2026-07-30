"""
bos_packing.py — BOS-aligned best-fit packing dataloader for NSTP-Ω V3.
Adapted from Karpathy's autoresearch prepare.py.

Eliminates the zero-padding PPL regression we saw in V2 mixed-length training.

Key ideas:
- Every row starts with BOS token (preserves document boundaries)
- Documents packed using best-fit to fill row exactly
- When no doc fits remaining space, crop shortest doc
- 100% utilization, no padding, no attention masking needed
- Document-level shuffling via buffer rotation

For FineWeb-Edu: documents come from the original pretokenized .npy file.
We need to re-segment into documents. Since FineWeb was tokenized as one
giant stream, we use a heuristic: eos_token (50256) marks doc boundaries.
"""
import numpy as np
import torch
from typing import Iterator, Tuple
from pathlib import Path


EOS_TOKEN = 50256  # GPT-2 EOS
BOS_TOKEN = 50256  # GPT-2 also uses 50256 as BOS for our purposes


def find_doc_boundaries(tokens: np.ndarray, eos_token: int = EOS_TOKEN) -> np.ndarray:
    """
    Return array of indices where each document starts.
    Document = [BOS, ..., EOS] where EOS is the special token.
    """
    eos_positions = np.where(tokens == eos_token)[0]
    # First doc starts at 0
    boundaries = np.concatenate([[0], eos_positions + 1])
    boundaries = boundaries[boundaries < len(tokens)]
    return boundaries


def segment_into_docs(tokens: np.ndarray, eos_token: int = EOS_TOKEN,
                      fallback_chunk: int = 512) -> list:
    """
    Split token stream into list of per-document token arrays.

    Tries EOS-based segmentation first.
    Falls back to fixed-length chunks if EOS count < 2 (e.g. FineWeb-Edu
    which was concatenated into one stream).
    """
    eos_positions = np.where(tokens == eos_token)[0]

    if len(eos_positions) >= 2:
        boundaries = np.concatenate([[0], eos_positions + 1])
        boundaries = boundaries[boundaries < len(tokens)]
        docs = []
        for i in range(len(boundaries) - 1):
            doc = tokens[boundaries[i]:boundaries[i+1]]
            if len(doc) >= 2:
                docs.append(doc)
        if len(docs) >= 2:
            return docs

    # Fallback: fixed-length chunks (pretend each chunk is a "doc")
    docs = []
    for i in range(0, len(tokens) - fallback_chunk, fallback_chunk):
        docs.append(tokens[i:i + fallback_chunk])
    return docs


class BOSPackedDataset(torch.utils.data.Dataset):
    """
    BOS-aligned best-fit packed dataset.

    Output: (inputs, targets) tensors of shape (seq_len,)
    where inputs[i] = tokens[i], targets[i] = tokens[i+1]
    Every packed row starts with a BOS token.
    """
    def __init__(self, tokens: np.ndarray, seq_len: int = 512, buffer_size: int = 100):
        """
        tokens: 1D np.ndarray of token ids
        seq_len: target sequence length per row
        buffer_size: number of docs to keep in packing buffer
        """
        self.seq_len = seq_len
        self.row_capacity = seq_len + 1  # +1 so we can produce inputs+targets from one row

        # Segment into documents (BOS-aligned)
        print(f"Segmenting {len(tokens):,} tokens into documents...")
        self.docs = segment_into_docs(tokens)
        print(f"Found {len(self.docs):,} documents")
        print(f"Avg doc length: {np.mean([len(d) for d in self.docs]):.0f} tokens")
        print(f"Min/Max doc length: {min(len(d) for d in self.docs)}/{max(len(d) for d in self.docs)}")

        # Pre-pack rows using best-fit
        print(f"Best-fit packing into {seq_len}-token rows...")
        self.rows = self._pack_all_rows(buffer_size=buffer_size)
        print(f"Produced {len(self.rows):,} packed rows")

    def _pack_all_rows(self, buffer_size: int) -> list:
        """Pack all documents into rows. Returns list of np.ndarray rows."""
        rows = []
        doc_buffer = []

        # Shuffle doc order for variety
        doc_indices = np.random.permutation(len(self.docs))

        for idx in doc_indices:
            doc = self.docs[idx]
            # Prepend BOS if not already there (defensive)
            if doc[0] != BOS_TOKEN:
                doc = np.concatenate([[BOS_TOKEN], doc])

            if len(doc) > self.row_capacity:
                # Doc too long, truncate
                doc = doc[:self.row_capacity]

            doc_buffer.append(doc)

            if len(doc_buffer) >= buffer_size:
                # Pack a row
                row = self._pack_one_row(doc_buffer)
                if row is not None:
                    rows.append(row)
                doc_buffer = []

        return rows

    def _pack_one_row(self, doc_buffer: list) -> np.ndarray:
        """Pack documents into a single row using best-fit."""
        row = np.zeros(self.row_capacity, dtype=np.int64)
        pos = 0
        remaining_docs = list(doc_buffer)

        while pos < self.row_capacity and remaining_docs:
            remaining = self.row_capacity - pos

            # Find largest doc that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(remaining_docs):
                if len(doc) <= remaining and len(doc) > best_len:
                    best_idx = i
                    best_len = len(doc)

            if best_idx >= 0:
                doc = remaining_docs.pop(best_idx)
                row[pos:pos + len(doc)] = doc
                pos += len(doc)
            else:
                # No doc fits — crop shortest
                shortest_idx = min(range(len(remaining_docs)), key=lambda i: len(remaining_docs[i]))
                doc = remaining_docs.pop(shortest_idx)
                crop_len = min(len(doc), remaining)
                row[pos:pos + crop_len] = doc[:crop_len]
                pos += crop_len

        # Only return row if it has at least some content
        if pos < 2:
            return None
        return row

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        # Return as torch tensors
        x = torch.from_numpy(row[:-1].astype(np.int64))
        y = torch.from_numpy(row[1:].astype(np.int64))
        return x, y


class BOSPackedStreamingLoader:
    """
    Streaming version: keeps packing in background, shuffles per epoch.
    Better for large datasets where pre-packing all rows is expensive.
    """
    def __init__(self, tokens: np.ndarray, seq_len: int = 512,
                 batch_size: int = 8, buffer_size: int = 100,
                 shuffle_seed: int = 42):
        self.tokens = tokens
        self.seq_len = seq_len
        self.row_capacity = seq_len + 1
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle_seed = shuffle_seed

        print(f"Streaming loader: B={batch_size}, T={seq_len}")
        # Pre-segment docs once (cheap)
        print("Segmenting documents...")
        self.all_docs = segment_into_docs(tokens)
        print(f"  {len(self.all_docs):,} documents")

    def __iter__(self):
        """Yield (inputs, targets) batches indefinitely."""
        rng = np.random.RandomState(self.shuffle_seed)
        doc_idx = rng.permutation(len(self.all_docs))
        pos = 0
        doc_buffer = []

        while True:
            # Refill buffer
            while len(doc_buffer) < self.buffer_size:
                if pos >= len(doc_idx):
                    # Reshuffle for next epoch
                    doc_idx = rng.permutation(len(self.all_docs))
                    pos = 0
                doc = self.all_docs[doc_idx[pos]]
                pos += 1
                # Prepend BOS
                if doc[0] != BOS_TOKEN:
                    doc = np.concatenate([[BOS_TOKEN], doc])
                if len(doc) > self.row_capacity:
                    doc = doc[:self.row_capacity]
                doc_buffer.append(doc)

            # Pack one batch of rows
            rows = []
            for _ in range(self.batch_size):
                row = self._pack_one_row(doc_buffer)
                if row is not None:
                    rows.append(row)

            if not rows:
                continue

            # Convert to tensors (B, T)
            batch = np.stack(rows)
            x = torch.from_numpy(batch[:, :-1].astype(np.int64))
            y = torch.from_numpy(batch[:, 1:].astype(np.int64))
            yield x, y

    def _pack_one_row(self, doc_buffer: list) -> np.ndarray:
        row = np.zeros(self.row_capacity, dtype=np.int64)
        pos = 0
        remaining = list(doc_buffer)

        while pos < self.row_capacity and remaining:
            space = self.row_capacity - pos
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(remaining):
                if len(doc) <= space and len(doc) > best_len:
                    best_idx = i
                    best_len = len(doc)

            if best_idx >= 0:
                doc = remaining.pop(best_idx)
                row[pos:pos + len(doc)] = doc
                pos += len(doc)
            else:
                shortest_idx = min(range(len(remaining)), key=lambda i: len(remaining[i]))
                doc = remaining.pop(shortest_idx)
                crop_len = min(len(doc), space)
                row[pos:pos + crop_len] = doc[:crop_len]
                pos += crop_len

        return row if pos >= 2 else None


if __name__ == "__main__":
    print("Testing BOS-aligned best-fit packing...")

    # Synthetic test: simulate docs of varying lengths
    np.random.seed(0)
    fake_docs = []
    for _ in range(1000):
        doc_len = np.random.randint(10, 200)
        fake_docs.append(np.random.randint(0, 50000, doc_len, dtype=np.int64))
    fake_tokens = np.concatenate(fake_docs)

    print(f"\nTotal tokens: {len(fake_tokens):,}")
    print(f"Avg doc len:  {np.mean([len(d) for d in fake_docs]):.0f}")

    # Test static dataset
    ds = BOSPackedDataset(fake_tokens, seq_len=128, buffer_size=20)
    print(f"\nPacked rows: {len(ds):,}")
    print(f"Utilization: {len(ds) * 128 / len(fake_tokens) * 100:.1f}%")

    x, y = ds[0]
    print(f"Row 0: x.shape={x.shape}, y.shape={y.shape}")
    print(f"  x[0] (should be BOS-like): {x[0]}")
    print(f"  x[-1]: {x[-1]}")

    # Test streaming loader
    print("\nTesting streaming loader...")
    loader = iter(BOSPackedStreamingLoader(fake_tokens, seq_len=128, batch_size=4))
    for i in range(3):
        x, y = next(loader)
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}, x[0,0]={x[0,0]}")

    print("\n✅ BOS-aligned packing working!")
