
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR, LambdaLR

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)



        
def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_type:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.00001,
            T_max=(args.epochs - args.warmup_epoch) * n_iter_per_epoch)
    elif "step" in args.lr_type:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=args.lr_decay_rate,
            milestones=[(m - args.warmup_epoch) * n_iter_per_epoch for m in args.lr_steps])        
        
    else:
        raise NotImplementedError("scheduler not supported:" + args.lr_type)


    if args.warmup_epoch != 0 :
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=args.warmup_epoch * n_iter_per_epoch)

    return scheduler