import torch

std = torch.Tensor([9.49077602, 2.49329645, 5.03377988, 1.76282567, 2.21402973, 2.97681828,
                    2.14723644, 0.6380834,  0.64613585, 0.99951063, 0.92877199, 6.6417578])
mean = torch.Tensor([25.60003559,  4.29149587,  9.21989404,  2.16928589,  2.85579326,  4.67167195,
                     2.49971274,  0.86798461,  0.53945552,  1.58682799,  2.31876199, 11.31548302])


def destandardize(output):
    output = torch.squeeze(output)
    for i in range(12):
        output[i] *= std
        output[i] += mean
    return output
