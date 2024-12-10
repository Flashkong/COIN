from fvcore.common.checkpoint import Checkpointer
from detectron2.engine.train_loop import HookBase
from typing import Optional,List,Any
from iopath.common.file_io import PathManager
from detectron2.engine.hooks import LRScheduler

class PeriodicCheckpointer:
    """
    Save checkpoints periodically. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, if iteration is a
    multiple of period or if `max_iter` is reached.

    Attributes:
        checkpointer (Checkpointer): the underlying checkpointer object
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
        **kwargs: Any
    ) -> None:
        """
        Args:
            checkpointer: the checkpointer object used to save checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "{file_prefix}_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
            file_prefix (str): the prefix of checkpoint's filename
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints: List[str] = []
        self.path_manager: PathManager = checkpointer.path_manager
        self.file_prefix = file_prefix
        self.addin = {}
        self.addin.update(kwargs)

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        
        if (iteration + 1) % self.period == 0:
            additional_state.update( 
                {k:(v.get_results() if hasattr(v,'get_results')  else v) for k,v in self.addin.items()}
            )
            self.checkpointer.save(
                "{}_{:07d}".format(self.file_prefix, iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.

        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)


class MyPeriodicCheckpointer(PeriodicCheckpointer, HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)


class MyLRScheduler(LRScheduler):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None, storage_name='lr'):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._storage_name = storage_name

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar(self._storage_name, lr, smoothing_hint=False)
        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler

from detectron2.engine.hooks import EvalHook
class MyEvalHook(EvalHook):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, eval_start=-1):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self._eval_start = eval_start

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            if next_iter > self._eval_start:
                # do the last eval in after_train
                if next_iter != self.trainer.max_iter:
                    self._do_eval()