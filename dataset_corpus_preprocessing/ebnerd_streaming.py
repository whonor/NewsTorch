"""Bounded-memory Parquet reads and disk-backed EB-NeRD behavior sequences."""

import os
import shutil
import tempfile
import weakref
from collections.abc import Sequence

import numpy as np


def iter_parquet_frames(path, columns=None, batch_size=4096):
    """Yield frames in physical row order, with global positional row indices."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            'EB-NeRD batch preprocessing requires pyarrow; install it with '
            '`python -m pip install pyarrow` in the training environment.'
        ) from error

    parquet = pq.ParquetFile(path)
    if columns is not None:
        columns = [name for name in columns if name in parquet.schema_arrow.names]
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        frame = batch.to_pandas()
        frame.index = range(offset, offset + len(frame))
        offset += len(frame)
        yield frame


BEHAVIOR_COLUMNS = (
    'uid', 'history', 'history_read_time', 'next_read_time',
    'next_scroll_percentage', 'labels', 'candidates',
)
NEWS_COLUMNS = (
    'nid', 'category', 'subcategory', 'title', 'abstract',
    'sentiment_label', 'sentiment_score', 'ner_clusters', 'topics', 'category_str',
)


def _cleanup_maps(directory, mappings):
    # NFS keeps an unlinked but still-open file as .nfs*, preventing rmdir.
    # Close mappings before unlinking their files, including at interpreter exit.
    for array in mappings:
        array._mmap.close()
    mappings.clear()
    shutil.rmtree(directory)


class BehaviorCache:
    """Shared lifetime for mapped files, retained by corpus and datasets."""

    def __init__(self, parent):
        self.name = tempfile.mkdtemp(prefix='behaviors-large-', dir=os.path.abspath(parent))
        self._mappings = []
        self._finalizer = weakref.finalize(self, _cleanup_maps, self.name, self._mappings)

    def track(self, array):
        if isinstance(array, np.memmap):
            self._mappings.append(array)
        return array

    def cleanup(self):
        self._finalizer()


class DiskBehaviors(Sequence):
    """Expose legacy behavior rows, storing histories once per impression.

    Files belong to the corpus's temporary directory. Only one accessed row is
    reconstructed; candidates and negatives remain in memory-mapped arrays.
    """

    def __init__(self, directory, mode, max_history_num, cache=None):
        self.mode = mode
        self.cache = cache
        self.context_dtype = np.dtype([
            ('user', '<i8'), ('history', '<i4', (max_history_num,)),
            ('history_length', '<i4'), ('read_time', '<f8', (max_history_num,)),
            ('negative_start', '<i8'), ('negative_count', '<i8'),
        ])
        self.sample_dtype = np.dtype([
            ('context', '<i8'), ('candidate', '<i4'), ('behavior', '<i8'),
            ('read_time', '<f8'), ('scroll', '<f8'),
        ])
        self.paths = [os.path.join(directory, mode + suffix)
                      for suffix in ('.contexts', '.samples', '.negatives')]
        self._files = [open(path, 'wb', buffering=1024 * 1024) for path in self.paths]
        self._last_behavior = None
        self._context_count = 0
        self._negative_count = 0

    def append(self, row):
        train = self.mode == 'train'
        behavior = int(row[5] if train else row[4])
        if behavior != self._last_behavior:
            negatives = np.asarray(row[4] if train else [], dtype='<i4')
            context = np.zeros(1, dtype=self.context_dtype)
            context['user'] = row[0]
            context['history'] = row[1]
            context['history_length'] = int(np.count_nonzero(row[2]))
            context['read_time'] = row[6] if train else row[5]
            context['negative_start'] = self._negative_count
            context['negative_count'] = len(negatives)
            self._files[0].write(context.tobytes())
            self._files[2].write(negatives.tobytes())
            self._negative_count += len(negatives)
            self._context_count += 1
            self._last_behavior = behavior
        sample = np.array([(
            self._context_count - 1, row[3], behavior,
            row[7] if train else row[6], row[8] if train else row[7],
        )], dtype=self.sample_dtype)
        self._files[1].write(sample.tobytes())

    def finalize(self):
        for handle in self._files:
            handle.close()
        del self._files
        arrays = []
        for path, dtype in zip(self.paths, (self.context_dtype, self.sample_dtype, np.dtype('<i4'))):
            count = os.path.getsize(path) // dtype.itemsize
            arrays.append(np.memmap(path, dtype=dtype, mode='r', shape=(count,))
                          if count else np.empty(0, dtype=dtype))
        self.contexts, self.samples, self.negatives = arrays
        if self.cache is not None:
            for array in arrays:
                self.cache.track(array)
        self.indices = self.samples['behavior']
        return self

    def training_candidates(self):
        """Iterate positive/negative IDs without reconstructing history rows."""
        for sample in self.samples:
            context = self.contexts[sample['context']]
            start = int(context['negative_start'])
            yield int(sample['candidate']), self.negatives[start:start + int(context['negative_count'])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        sample = self.samples[index]
        context = self.contexts[sample['context']]
        mask = np.zeros(len(context['history']), dtype=np.float32)
        mask[:context['history_length']] = 1
        row = [int(context['user']), context['history'], mask, int(sample['candidate'])]
        if self.mode == 'train':
            start = int(context['negative_start'])
            row.append(self.negatives[start:start + int(context['negative_count'])])
        row.extend([int(sample['behavior']), context['read_time'],
                    float(sample['read_time']), float(sample['scroll'])])
        return row
