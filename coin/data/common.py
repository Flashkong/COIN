import logging
from detectron2.data.common import AspectRatioGroupedDataset

class AspectRatioGroupedDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        self._buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            w, h = d[0]["width"], d[0]["height"]
            bucket_id = 0 if w > h else 1

            # bucket = bucket for normal images
            bucket = self._buckets[bucket_id]
            bucket.append(d[0])

            # buckets_key = bucket for augmented images
            buckets_key = self._buckets_key[bucket_id]
            buckets_key.append(d[1])
            if len(bucket) == self.batch_size:
                yield (bucket[:], buckets_key[:])
                del bucket[:]
                del buckets_key[:]