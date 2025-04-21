import time
from typing import List, Optional, Union

from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated
import pdb
## added by lsun
from plot_lsun import plot
from pathlib import Path
class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        wandb_mode: str = "disabled",
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        
        print('use modified trainer')
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        wandb.init(mode=wandb_mode, **kwargs)
        

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        wandb.watch(net, log="all", log_freq=self.num_batches_per_epoch)

        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )
        # added by lsun
        print('start to train')
        #pdb.set_trace()
        #
        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            avg_epoch_loss = 0.0

            #with tqdm(train_iter) as it:
            it = train_iter

            # temp0 = next(iter(it))
            # for k,v in temp0.items():
            #     print(k+' shape is', v.shape)
            # pdb.set_trace()

            for batch_no, data_entry in enumerate(it, start=1):
                #print('batch_no is', batch_no)
                optimizer.zero_grad()
                inputs = [v.to(self.device) for v in data_entry.values()]

                # for k,v in data_entry.items():
                #      print(k+' shape is', v.shape)

                #print('stop at traininer')
                #pdb.set_trace()
                output = net(*inputs)
                if isinstance(output, (list, tuple)):
                    loss = output[0]
                else:
                    loss = output

                avg_epoch_loss += loss.item()
                # it.set_postfix(
                #     ordered_dict={
                #         "avg_epoch_loss": avg_epoch_loss / batch_no,
                #         "epoch": epoch_no,
                #     },
                #     refresh=False,
                # )
                if batch_no%10 == 0:
                    print('epoch: ', str(epoch_no), 'batch ', str(batch_no),'average_epoch_loss :', avg_epoch_loss / batch_no)

                #pdb.set_trace()
                # ## added by lsun
                # if batch_no == 1 and epoch_no%1 == 0:
                #     pdb.set_trace()
                #     savedir = 'plot/test_gluon07_test/'
                #     #Path(savedir).mkdir(parents = True, exist_ok = True)
                #     fname = savedir+'train_ep_'+str(epoch_no)+'.png'
                #     # %%
                #     plot(
                #         target=tss[0],
                #         forecast=forecasts[0],
                #         prediction_length=metadata['prediction_length'],
                #         fname=fname
                #     )

                # ##
                wandb.log({"loss": loss.item()})

                loss.backward()
                if self.clip_gradient is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                optimizer.step()
                lr_scheduler.step()

                if self.num_batches_per_epoch == batch_no:
                    break

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

        # writer.close()
