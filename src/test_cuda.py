import torch

def test_cuda():
    a = torch.arange(101).to('cuda')
    assert (a * a)[100].item() == 10000

if __name__ == '__main__':
    test_cuda()
