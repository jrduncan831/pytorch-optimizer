import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params


def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

def _hessian_linear_systems(data: torch.Tensor) -> torch.Tensor:
    return 4* (data.t() @ data) / (data.size()[0])

def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return 2*hessian.cpu().data

class Newtons_Method(Optimizer):
    r"""Implements Newtons Method Optimizer. This implementation is meant for smaller systems. 
    Hessian matrix intractable in high dimensions

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Shampoo(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1802.09568

    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
        write_hessian: bool = True,
        hessian_file: str = 'hessian.txt'
    ):
        self.write_hessian = write_hessian
        self.hessian_file = hessian_file

        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if update_freq < 1:
            raise ValueError('Invalid momentum value: {}'.format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(Newtons_Method, self).__init__(params, defaults)

    def step(self, weights, model, loss_grad, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:   # looks over parameters
            for p in group['params']:     # getting gradients -- NEED TO LOOK AT MULTILAYER NETWORKS, STRUCTURE
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                if len(state) == 0:
                    dim = grad.size()[1] 
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                #    state['precond'] = group[
                #            'epsilon'
                #        ] * torch.eye(dim, out=grad.new(grad.size(), dim))

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(
                        state['momentum_buffer'], alpha=momentum
                    )

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # See Algorithm 2 for detail

                precond = _hessian_linear_systems((weights))  #state['precond']
                print('hessian linear:',precond)
                precond = eval_hessian(loss_grad,model)
                print('hessian:',precond) 
           #     precond.add_( _hessian_linear_systems((p)) ) # WORKING HERE FIX!!!!!
                inv_precond = torch.linalg.inv(precond) #_matrix_power(precond, -1)
                print('inv hessian:',inv_precond) 
                grad = (inv_precond @ grad.t()).t()
                print('inv hessian @ grad:',grad) 
                state['step'] += 1
                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-group['lr'])

                if self.write_hessian:
                    with open('hessian.txt','a+') as f:
                        f.write('Step Number: {}'.format(str(state['step'])))
                        f.write(' Final Search Vector:' + str(grad.numpy().tolist()))
                        f.write('Hessian:' + str(precond.numpy().tolist()))
                        f.write('Inverse Hessian:' + str(inv_precond.numpy().tolist()))
                        f.write('\n')
                state['step'] += 1
                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-group['lr'])

        return loss
