import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


# weight 是一个以guess为均值，1为标准差的正太分布
# measurement 是一个以weight为均值，0.75为标准差的正太分布（注，weight是个分布）
def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))


# 用来拟合分布的分布
def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))

# scale 这个分布的条件分布，条件为measurement = 9.5（注，P[scale|measurement = 9.5]）
conditioned_scale = pyro.condition(scale, data={"measurement": torch.tensor(9.5)})

guess = 8.5

# 拟合方法设置
pyro.clear_param_store()
svi = pyro.infer.SVI(model=conditioned_scale,
                     guide=scale_parametrized_guide,
                     optim=pyro.optim.Adam({"lr": 0.003}),
                     loss=pyro.infer.Trace_ELBO())

# 训练并保存中间结果
losses, a, b = [], [], []
num_steps = 8500
for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

plt.figure(1)
plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
print('a = ', pyro.param("a").item())
print('b = ', pyro.param("b").item())

plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot([0, num_steps], [9.14, 9.14], 'k:')
plt.plot(a)
plt.ylabel('a')

plt.subplot(1, 2, 2)
plt.ylabel('b')
plt.plot([0, num_steps], [0.6, 0.6], 'k:')
plt.plot(b)
plt.tight_layout()
plt.show()
