from typing import NamedTuple, Optional
from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader

from gluonts.env import env
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.estimator import Estimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import SelectFields, Transformation
#from gluonts.itertools import maybe_len
from missingutils import maybe_len

#from pts import Trainer
from trainer import Trainer
from missingutils import get_module_forward_input_names
#from pts.model import get_module_forward_input_names
from localpts.dataset.loader import TransformedIterableDataset

import pdb
class TrainOutput(NamedTuple):
    print('whats this')
    #pdb.set_trace()

    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor


class PyTorchEstimator(Estimator):
    @validated()
    def __init__(
        self, trainer: Trainer, lead_time: int = 0, dtype: np.dtype = np.float32
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.trainer = trainer
        self.dtype = dtype

    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        raise NotImplementedError

    def create_instance_splitter(self, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        raise NotImplementedError

    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        nn.Module
            The network that computes the loss given input data.
        """
        raise NotImplementedError

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> PyTorchPredictor:
        """
        Create and return a predictor object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        raise NotImplementedError

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> TrainOutput:
        #pdb.set_trace()
        print('calling the train model')
        transformation = self.create_transformation()

        trained_net = self.create_training_network(self.trainer.device)

        input_names = get_module_forward_input_names(trained_net)

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            training_instance_splitter = self.create_instance_splitter("training")
        ## added by lsun
        # temp0 = next(iter(training_data))
        # for k,v in temp0.items():
        #     print('k is', k)
        #     if k != 'start' and k!='source':
        #         #continue
        #         print(k+' shape is', v.shape)
        # print('before transformation')
        # print('\n')
        ## end added
        training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=transformation
            + training_instance_splitter
            + SelectFields(input_names),
            is_train=True,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
        )
        ## added by lsun
        # temp1 = next(iter(training_iter_dataset))
        # for k,v in temp1.items():
        #     print(k+' shape is', v.shape)
        #     np.savez(k,v = v)
        # print('after transformation')
        # print('\n')
        # pdb.set_trace()
        ## end add
        ###
        ## here we use batchsize
        ####
        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            **kwargs,
        )
        ## added by lsun
        #dict_key = []
        #dict_values = []

        # temp2 = next(iter(training_data_loader))
        # for k,v in temp2.items():
        #     print(k+' shape is', v.shape)
        #     #np.savez(k,v = v)

        # pdb.set_trace()
        ## end add
        ##
        validation_data_loader = None
        if validation_data is not None:
            with env._let(max_idle_transforms=maybe_len(validation_data) or 0):
                validation_instance_splitter = self.create_instance_splitter("validation")
            validation_iter_dataset = TransformedIterableDataset(
                dataset=validation_data,
                transform=transformation
                + validation_instance_splitter
                + SelectFields(input_names),
                is_train=True,
                cache_data=cache_data,
            )
            validation_data_loader = DataLoader(
                validation_iter_dataset,
                batch_size=self.trainer.batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn,
                **kwargs,
            )
        ## 
        print('stop before calling traininer')
        #pdb.set_trace()
        self.trainer(
            net=trained_net,
            train_iter=training_data_loader,
            validation_iter=validation_data_loader,
        )
        print('before TrainOutput')
        #pdb.set_trace()
        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> PyTorchPredictor:
        return self.train_model(
            training_data,
            validation_data,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            **kwargs,
        ).predictor
