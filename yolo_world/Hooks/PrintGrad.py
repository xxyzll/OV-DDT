from mmengine.hooks import Hook
from mmyolo.registry import HOOKS
from typing import Dict, Optional, Sequence, Union

@HOOKS.register_module()
class GradientPrintingHook(Hook):
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch,
                         outputs: Optional[dict] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        pass
        # runner.model.embeddings
        # print(f' gradients') 
        