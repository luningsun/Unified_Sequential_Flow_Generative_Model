from typing import Optional
import itertools

from torch.utils.data import IterableDataset

from gluonts.dataset.common import Dataset
from gluonts.transform import Transformation, TransformedDataset
from gluonts.itertools import Cyclic, PseudoShuffled, Cached

import pdb
class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length

        # temp0 = next(iter(dataset))
        # print('keys before loader transfomration')
        # print('\n')
        # print(temp0.keys())
        # print('stop at the transformed dataset')
        # for k,v in temp0.items():
        #     if k != 'start' and k!='source':
        #         #pdb.set_trace()
        #         print(k+' shape is', v.shape)
        #pdb.set_trace()
        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if not cache_data else Cached(Cyclic(dataset)),
            transform,
            is_train=is_train,
        )
        # temp2 = next(iter(self.transformed_dataset))
        # print('keys after loader transfomration')
        # print('\n')
        # print(temp2.keys())
        # print('shape of the dict')
        # for k,v in temp2.items():
        #     print(k+' shape is', v.shape)
        #pdb.set_trace()
    def __iter__(self):
        print('calling the dataset')
        #print('self.shuffle_buffer_length is', self.shuffle_buffer_length)
        #pdb.set_trace()
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        else:
            shuffled = PseudoShuffled(
                self.transformed_dataset,
                shuffle_buffer_length=self.shuffle_buffer_length,
            )
            return iter(shuffled)
