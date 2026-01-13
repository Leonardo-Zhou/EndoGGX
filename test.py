import datasets
import os

dataset = datasets.HamlynDataset(
    data_path='/data1/publicData/hamlyn_data',
    height=256,
    width=320,
    frame_idxs=[0],
    num_scales=4,
    is_train=False
)

print(dataset.scans)