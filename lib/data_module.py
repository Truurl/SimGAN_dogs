
import pytorch_lightning as L
import config as cfg
import torch
import torch.utils.data as Data
import h5py
from PIL import Image
import numpy as np

class HDF5Dataset(torch.utils.data.Dataset):
    

    def __init__(self, file_path, datasets: tuple, transform=None):
        super().__init__()

        self.file_path = file_path
        self.datasets = datasets

        self.transform = transform

        with h5py.File(self.file_path) as file:
            self.length = len(file[self.datasets[0]])
            self.target = file[self.datasets[1]][0]
            # print(f'self.target: {self.target}')

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index) :
        if not hasattr(self, '_hf'):
            self._open_hdf5()
        # print(index)
        # with h5py.File(self.file_path, 'r') as file:

        #     x = file[self.datasets[0]][index]
        #     x = Image.fromarray(np.array(x))

        #     if self.transform:
        #         x = self.transform(x)
        #     else:
        #         x = torch.from_numpy(x)

        #     y = file[self.datasets[1]][index]
        #     target = torch.zeros(1,2)
        #     target = target[:, y] = 1

        # return (x, y)
        # get data
        x = self._hf[self.datasets[0]][index]
        x = Image.fromarray(np.array(x))
        # x = self.data[index]
        # x = Image.fromarray(np.array(x))
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # y = self._hf[self.datasets[1]][index]
        y = self.target
        target = torch.zeros(1,2)
        y = target[:, y] = 1
        return (x, y)

    def __len__(self):
        return self.length
    
class EyeDatasetModule(L.LightningDataModule):

    def __init__(self, synth_path, synth_datasets, real_path, real_datasets, transform=None, num_workers: int = 4, batch_size: int = 32):
        super().__init__()
        self.synth_path = synth_path
        self.synth_datasets = synth_datasets
        self.real_path = real_path
        self.real_datasets = real_datasets

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.transform = transform

    def setup(self, stage="fit") -> None:

        self.synth_dataset = HDF5Dataset(self.synth_path, self.synth_datasets, transform=self.transform)
        self.real_dataset = HDF5Dataset(self.real_path, self.real_datasets, transform=self.transform)

    def train_dataloader(self) -> Data.DataLoader:
        return {'synth': Data.DataLoader(self.synth_dataset, self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers),
                'real': Data.DataLoader(self.real_dataset, self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)}

    
    def val_dataloader(self) -> Data.DataLoader:
        # synth_dataset = HDF5Dataset(self.synth_path, self.synth_datasets, transform=self.transform)
        # real_dataset = HDF5Dataset(self.real_path, self.real_datasets, transform=self.transform)
        iterables = {'synth': Data.DataLoader(self.synth_dataset, self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers),
                'real': Data.DataLoader(self.real_dataset, self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)}
        # print(synth_dataset, real_dataset)
        return L.trainer.supporters.CombinedLoader(iterables)
        # return {'synth': Data.DataLoader(synth_dataset, self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers),
        #         'real': Data.DataLoader(real_dataset, self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)}
