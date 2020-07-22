import chainer
import numpy
import torch

import chainer_pytorch_migration as cpm


def test_one_output():
    torch_fn = torch.sigmoid
    x = chainer.Variable(numpy.ones((5, 5), dtype=numpy.float32))
    z = chainer.functions.sin(x)
    res = cpm.chainer_torch_function(torch_fn, z)
    res = chainer.functions.sqrt(res)
    res = cpm.chainer_torch_function(torch_fn, res)
    res = chainer.functions.sqrt(res)
    res.grad = numpy.ones((5, 5), dtype=numpy.float32)
    res.backward()
    c_grad = x.grad

    # Do it now in pytorch and compare
    x = torch.ones((5, 5), requires_grad=True)
    z = torch.sin(x)
    y = torch.sigmoid(torch.sigmoid(z).sqrt()).sqrt()
    y.backward(torch.ones(5, 5))
    t_grad = x.grad
    assert torch.allclose(t_grad, cpm.astensor(c_grad))


def test_multiple_outputs():
    torch_fn = torch.split
    x = chainer.Variable(numpy.ones((6, 5), dtype=numpy.float32))
    y = chainer.functions.sin(x)
    y, z = cpm.chainer_torch_function(torch_fn, y, 3, dim=0)
    y = chainer.functions.log(y)
    z = chainer.functions.cos(z)
    z = y + z
    z.grad = numpy.ones((3, 5), dtype=numpy.float32)
    z.backward()
    c_grad = x.grad

    x = torch.ones((6, 5), requires_grad=True)
    z = torch.sin(x)
    y, z = torch.split(z, 3, dim=0)
    y = torch.log(y)
    z = torch.cos(z)
    z = y + z
    z.backward(torch.ones((3, 5)))
    t_grad = x.grad
    assert torch.allclose(t_grad, cpm.astensor(c_grad))


def test_torch_chainer_function():
    class TorchChainerSigmoid(cpm.functions.TorchChainerFunction):
        @staticmethod
        def chainer_fn():
            return chainer.functions.sigmoid
    # Combined torch
    x = torch.ones(10)
    x.requires_grad = True
    y = torch.sin(x)
    y = TorchChainerSigmoid.apply(y)
    y = torch.sum(y)
    y.backward()
    ct_grad = x.grad

    # All in torch
    x = torch.ones(10)
    x.requires_grad = True
    y = torch.sin(x)
    y = torch.sigmoid(y)
    y = torch.sum(y)
    y.backward()
    assert torch.allclose(ct_grad, x.grad)


def test_torch_chainer_function_2():
    class TorchChainerAdd(cpm.functions.TorchChainerFunction):
        @staticmethod
        def chainer_fn():
            return chainer.functions.add
    # Combined torch
    a = torch.ones(10)
    a.requires_grad = True
    b = torch.ones(10)+2
    b.requires_grad = True
    y = torch.sin(a)
    z = torch.sin(b)
    y = TorchChainerAdd.apply(y, z)
    y = torch.sum(y)
    y.backward()
    a_ct_grad = a.grad
    b_ct_grad = b.grad

    # All in torch
    a = torch.ones(10)
    a.requires_grad = True
    b = torch.ones(10)+2
    b.requires_grad = True
    y = torch.sin(a)
    z = torch.sin(b)
    y = torch.add(y, z)
    y = torch.sum(y)
    y.backward()
    assert torch.allclose(a_ct_grad, a.grad)
    assert torch.allclose(b_ct_grad, b.grad)
