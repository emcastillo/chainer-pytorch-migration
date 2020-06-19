import chainer
import torch

import chainer_pytorch_migration as cpm


class _ChainerTorchFunction(chainer.FunctionNode):
    def __init__(self, torch_fn, *args, **kwargs):
        self.torch_fn = torch_fn
        self.torch_fwd_inputs = None
        self.torch_fwd_outputs = None
        self.args = args
        self.kwargs = kwargs

    def forward(self, inputs):
        t_inputs = [cpm.astensor(x) for x in inputs]
        for t_x in t_inputs:
            t_x.requires_grad = True
        self.torch_fwd_inputs = t_inputs
        f_inputs = t_inputs + list(self.args)
        # The torch function might require other arguments other than input
        # tensors so append them here
        t_outs = self.torch_fn(*f_inputs, **self.kwargs)
        if type(t_outs) is not list and type(t_outs) is not tuple:
            t_outs = (t_outs,)
        self.torch_fwd_outputs = t_outs
        # Need to access res from a chainer variable
        c_outs = tuple(cpm.asarray(out) for out in t_outs)
        # The outputs will be used in the grad function so we should retain
        # them ?
        self.retain_outputs(tuple(range(len(c_outs))))
        return c_outs

    def backward(self, indexes, grads):
        out_grads = _ChainerTorchFunctionGrad(
            self.torch_fwd_inputs, self.torch_fwd_outputs
        ).apply(grads)
        return out_grads


class _ChainerTorchFunctionGrad(chainer.FunctionNode):
    def __init__(self, inputs, outputs):
        super(_ChainerTorchFunctionGrad, self).__init__()
        self.inputs = inputs
        self.outputs = outputs

    def forward(self, inputs):
        t_grads = tuple([cpm.astensor(g) for g in inputs])
        torch.autograd.backward(self.outputs, t_grads)
        out_grads = tuple(
            cpm.asarray(x.grad) for x in self.inputs
        )
        self.outputs = [x.grad for x in self.inputs]
        self.inputs = t_grads
        return out_grads

    def backward(self, indexes, grads):
        return _ChainerTorchFunctionGrad(
            self.inputs, self.outputs).apply(grads)


def chainer_torch_function(torch_fn, inputs, *args, **kwargs):
    if type(inputs) is not list and type(inputs) is not tuple:
        inputs = (inputs,)
    y = _ChainerTorchFunction(torch_fn, *args, **kwargs).apply(inputs)
    if len(y) == 1:
        return y[0]
    return y
