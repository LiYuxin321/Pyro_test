import os
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
# Pyro also has a reparameterized Beta distribution so we import
# the non-reparameterized version to make our point
from pyro.distributions.testing.fakes import NonreparameterizedBeta
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
import sys

# assert pyro.__version__.startswith('1.7.0')

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
max_steps = 2 if smoke_test else 10000


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).item()


class BernoulliBetaExample:
    def __init__(self, max_steps):
        # the maximum number of inference steps we do
        self.max_steps = max_steps
        # the two hyperparameters for the beta prior
        self.alpha0 = 10.0
        self.beta0 = 10.0
        # the dataset consists of six 1s and four 0s
        self.data = torch.zeros(10)
        self.data[0:6] = torch.ones(6)
        self.n_data = self.data.size(0)
        # compute the alpha parameter of the exact beta posterior
        self.alpha_n = self.data.sum() + self.alpha0
        # compute the beta parameter of the exact beta posterior
        self.beta_n = - self.data.sum() + torch.tensor(self.beta0 + self.n_data)
        # initial values of the two variational parameters
        self.alpha_q_0 = 15.0
        self.beta_q_0 = 15.0

    def model(self, use_decaying_avg_baseline):
        # sample `latent_fairness` from the beta prior
        f = pyro.sample("latent_fairness", dist.Beta(self.alpha0, self.beta0))
        # use plate to indicate that the observations are
        # conditionally independent given f and get vectorization
        with pyro.plate("data_plate"):
            # observe all ten datapoints using the bernoulli likelihood
            pyro.sample("obs", dist.Bernoulli(f), obs=self.data)

    def guide(self, use_decaying_avg_baseline):
        # register the two variational parameters with pyro
        alpha_q = pyro.param("alpha_q", torch.tensor(self.alpha_q_0),
                             constraint=constraints.positive)
        beta_q = pyro.param("beta_q", torch.tensor(self.beta_q_0),
                            constraint=constraints.positive)
        # sample f from the beta variational distribution
        baseline_dict = {'use_decaying_avg_baseline': use_decaying_avg_baseline,
                         'baseline_beta': 0.90}
        # note that the baseline_dict specifies whether we're using
        # decaying average baselines or not
        pyro.sample("latent_fairness", NonreparameterizedBeta(alpha_q, beta_q),
                    infer=dict(baseline=baseline_dict))

    def do_inference(self, use_decaying_avg_baseline, tolerance=0.80):
        # clear the param store in case we're in a REPL
        pyro.clear_param_store()
        # setup the optimizer and the inference algorithm
        optimizer = optim.Adam({"lr": .0005, "betas": (0.93, 0.999)})
        svi = SVI(self.model, self.guide, optimizer, loss=TraceGraph_ELBO())
        print("Doing inference with use_decaying_avg_baseline=%s" % use_decaying_avg_baseline)

        # do up to this many steps of inference
        for k in range(self.max_steps):
            svi.step(use_decaying_avg_baseline)
            if k % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            # compute the distance to the parameters of the true posterior
            alpha_error = param_abs_error("alpha_q", self.alpha_n)
            beta_error = param_abs_error("beta_q", self.beta_n)

            # stop inference early if we're close to the true posterior
            if alpha_error < tolerance and beta_error < tolerance:
                break

        print("\nDid %d steps of inference." % k)
        print(("Final absolute errors for the two variational parameters " +
               "were %.4f & %.4f") % (alpha_error, beta_error))

# do the experiment
bbe = BernoulliBetaExample(max_steps=max_steps)
bbe.do_inference(use_decaying_avg_baseline=True)
bbe.do_inference(use_decaying_avg_baseline=False)
