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


class TorchChainerFunction(torch.autograd.Function):
    @staticmethod
    def chainer_fn():
        raise RuntimeError('chainer_fn function must be overriden')

    @classmethod
    def forward(cls, ctx, *inputs):
        chainer_fn = cls.chainer_fn()
        ctx.save_for_backward(*inputs)
        c_inputs = tuple((chainer.Variable(cpm.asarray(x)) for x in inputs))
        ctx.c_inputs = c_inputs
        c_outputs = chainer_fn(*c_inputs)
        if not type(c_outputs) is tuple:
            c_outputs = (c_outputs,)
        t_outputs = [cpm.astensor(y.array) for y in c_outputs]
        for t_y in t_outputs:
            t_y.requires_grad = True
        ctx.c_outputs = c_outputs
        if len(t_outputs) == 1:
            return t_outputs[0]
        else:
            return tuple(t_outputs)

    @staticmethod
    def backward(ctx, *grads):
        grads = [ctx.c_outputs, ctx.c_inputs] + list(grads)
        out_grads = _TorchChainerFunctionGrad.apply(*grads)
        return out_grads


class _TorchChainerFunctionGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *inputs):
        c_outputs = inputs[0]
        c_inputs = inputs[1]
        inputs = inputs[2:]
        ctx.save_for_backward(*inputs)
        c_grads = tuple((chainer.Variable(cpm.asarray(g)) for g in inputs))
        fwd_outputs = c_outputs
        chainer.backward(fwd_outputs, c_grads, enable_double_backprop=True)
        out_grads = tuple(
            cpm.astensor(x.grad) for x in c_inputs
        )
        for t_y in out_grads:
            t_y.requires_grad = True
        ctx.c_outputs = [x.grad for x in c_inputs]
        ctx.c_inputs = c_grads
        return out_grads

    def backward(ctx, *grads):
        grads = [ctx.c_outputs, ctx.c_inputs] + list(grads)
        return _TorchChainerFunctionGrad.apply(*grads)
