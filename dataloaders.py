from glob import glob
import os

from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.types as types
import nvidia.dali.fn as fn


@pipeline_def
def video_pipeline(filenames, sequence_length, size, initial_fill=1024,
                   shard_id=0, num_shards=1, random_shuffle=True,
                   reader_name='VideoReader', random_seed=12345):
    # Get video readers
    videos = fn.readers.video_resize(
        device='gpu', filenames=filenames, sequence_length=sequence_length,
        size=size, random_shuffle=random_shuffle, initial_fill=initial_fill,
        pad_last_batch=True, shard_id=shard_id, num_shards=num_shards,
        dtype=types.FLOAT, seed=random_seed, name=reader_name
    )

    return videos


class DALIVideoLoader():
    def __init__(
        self, video_root, batch_size, rank, local_rank, world_size,
        sequence_length=10, size=[320, 576], initial_fill=1024, num_workers=2,
        random_shuffle=True, reader_name='VideoReader', auto_reset=False,
        random_seed=12345
    ):
        filenames = glob(os.path.join(video_root, "*.mp4"))
        self.pipeline = video_pipeline(
            batch_size=batch_size,
            filenames=filenames,
            sequence_length=sequence_length,
            size=size,
            num_threads=num_workers,
            device_id=local_rank,
            initial_fill=initial_fill,
            shard_id=rank,
            num_shards=world_size,
            reader_name=reader_name,
            random_shuffle=random_shuffle,
            random_seed=random_seed
        )
        self.pipeline.build()
        self.dali_iterator = DALIGenericIterator(
            self.pipeline,
            ['sequence'],
            reader_name=reader_name,
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=auto_reset
        )

    def __iter__(self):
        return self.dali_iterator.__iter__()


@pipeline_def
def sequence_pipeline(image_dir, sequence_length, size,
                      initial_fill=1024, random_shuffle=True,
                      random_seed=12345):
    sequences = fn.readers.sequence(
        file_root=image_dir, sequence_length=sequence_length,
        initial_fill=initial_fill, random_shuffle=random_shuffle,
        seed=random_seed
    )
    sequences = fn.resize(sequences, size=size)
    return sequences


class DALISequenceLoader():
    def __init__(
        self, image_dir, sequence_length, batch_size, size=[320, 576],
        local_rank=0, initial_fill=1024, num_workers=2, random_shuffle=True,
        auto_reset=False, random_seed=12345
    ):
        self.pipeline = sequence_pipeline(
            image_dir=image_dir,
            sequence_length=sequence_length,
            size=size,
            initial_fill=initial_fill,
            batch_size=batch_size,
            device_id=local_rank,
            num_threads=num_workers,
            random_shuffle=random_shuffle,
            random_seed=random_seed
        )
        self.pipeline.build()
        self.dali_iterator = DALIGenericIterator(
            self.pipeline,
            ['sequence'],
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=auto_reset
        )

    def __iter__(self):
        return self.dali_iterator.__iter__()
