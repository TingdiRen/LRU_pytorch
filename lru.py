import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LRU(nn.Module):
    def __init__(self, in_features, activation=torch.relu, r_min=0.9, r_max=0.999, use_bias=True,
                 unroll=False):
        super(LRU, self).__init__()
        self.hidden_size = in_features
        self.activation = activation
        self.use_bias = use_bias
        self.unroll = unroll  # The parallel algorithm will divide and conquer more if True

        self.i_dense = nn.Linear(in_features, in_features * 2, bias=use_bias)  # Extend to the complex C
        self.o_dense = nn.Linear(in_features * 2, in_features, bias=use_bias)  # Back to real R

        # Initialize parameters
        u1 = np.random.random(size=in_features)
        u2 = np.random.random(size=in_features)
        v_log = np.log(-0.5 * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))  # defined in [arxiv] lemma 3.2
        theta_log = np.log(u2 * np.pi * 2)  # defined in [arxiv] lemma 3.2
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(v_log)) ** 2))  # defined above eq.7 of [arxiv]

        # defined in Optimization under exponential parameterization of [arxiv] 3.3
        self.params_log = nn.Parameter(torch.tensor([v_log, theta_log, gamma_log], dtype=torch.float32))

    def lru_parallel(self, i, x, v, theta, B, L, D):
        # Upper/low parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        l = 2 ** i
        x = x.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        x1, x2 = x[:, :l // 2], x[:, l // 2:]  # Divide the data in half

        pos = torch.arange(1, l // 2 + 1, dtype=torch.float, device=x.device)  # t=k+1 ~ T
        vs = torch.einsum('n,d->nd', pos, v)
        thetas = torch.einsum('n,d->nd', pos, theta)
        lambs = torch.exp(
            torch.complex(-vs, thetas))  # defined in Optimization under exponential parameterization of [arxiv] 3.3

        x2 = x2 + (lambs * x1[:, -1:])  # Add the last element of the half to the second half
        x = torch.cat([x1, x2], axis=1)
        if (not self.unroll) and x.shape[1] is not None:
            x = x.reshape(B, L, D)

        return i + 1, x, v, theta, B, L, D

    def while_loop(self, cond, body, loop_vars):
        while cond(*loop_vars[:2]):
            loop_vars = body(*loop_vars)
        return loop_vars

    def forward(self, inputs):
        u = self.i_dense(inputs)
        params = torch.exp(self.params_log)
        v, theta, gamma = params[0], params[1], params[2]

        len_seq_in = u.size(1)
        log2_L = int(np.ceil(np.log2(len_seq_in)))

        u = torch.view_as_complex(u.view(u.size(0), u.size(1), u.size(2) // 2, 2))
        u = F.pad(u,
                  (0, 0, 0, 2 ** log2_L - u.size(1), 0, 0))  # pad the sequence length to the power of 2 (for algorithm)
        B, L, D = u.size(0), u.size(1), u.size(2)

        if self.unroll:
            x = u  # init hidden states as inputs
            for i in range(log2_L):
                _, x, *_ = self.lru_parallel(i + 1, x, v, theta, B, L, D)
        else:
            _, x, *_ = self.while_loop(lambda i, x: i <= log2_L, self.lru_parallel, [1, u, v, theta, B, L, D])

        x = x[:, :len_seq_in] * (gamma.to(torch.complex64) + 0j)  # Element-wise parameter defined in [arxiv] eq.(7)
        x = self.complex_to_real_imag(x)
        output = self.o_dense(x)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def complex_to_real_imag(self, x):
        real_x = torch.real(x)
        imag_x = torch.imag(x)
        return torch.cat((real_x, imag_x), dim=-1)

    @torch.no_grad()
    def infer_step(self, input_t, hidden_state_t_1=None):
        u_t = self.i_dense(input_t)
        u_t = torch.view_as_complex(u_t.view(u_t.size(0), u_t.size(1), u_t.size(2) // 2, 2))
        params = torch.exp(self.params_log)
        v, theta, gamma = params[0], params[1], params[2]

        if hidden_state_t_1 is None:
            x_t = u_t
        else:
            x_t_1 = hidden_state_t_1
            lamb = torch.exp(torch.complex(-v, theta))
            x_t = lamb * x_t_1 + u_t
        y_t = x_t * (gamma.to(torch.complex64) + 0j)  # Element-wise parameter defined in [arxiv] eq.(7)
        y_t = self.complex_to_real_imag(y_t)
        y_t = self.o_dense(y_t)
        if self.activation is not None:
            y_t = self.activation(y_t)
        return x_t, y_t

    @torch.no_grad()
    def infer_steps(self, input_t, hidden_state_t_1=None):
        u_t = self.i_dense(input_t)
        u_t = torch.view_as_complex(u_t.view(u_t.size(0), u_t.size(1), u_t.size(2) // 2, 2))
        params = torch.exp(self.params_log)
        v, theta, gamma = params[0], params[1], params[2]

        x_t_1 = torch.zeros((u_t.shape[0], u_t.shape[2])) if hidden_state_t_1 is None else hidden_state_t_1[:, -1]
        lamb = torch.exp(torch.complex(-v, theta))
        x_t = torch.zeros_like(u_t)
        for i in range(u_t.shape[1]):
            x_t_t = lamb * x_t_1 + u_t[:, i]
            x_t_1 = x_t_t

            x_t[:, i, :] = x_t_t

        y_t = x_t * (gamma.to(torch.complex64) * 1j)  # Element-wise parameter defined in [arxiv] eq.(7)
        y_t = self.complex_to_real_imag(y_t)
        y_t = self.o_dense(y_t)
        if self.activation is not None:
            y_t = self.activation(y_t)
        return x_t, y_t

if __name__ == '__main__':
    torch.manual_seed(42)
    config = {'in_features': 53,
              'unroll': True}
    model = LRU(**config)
    model.eval()
    B, T, D = 11, 15, config['in_features']
    u = torch.randn((B, T, D))
    # parallel infer
    y = model(u)
    # Serial infer
    x1, y1 = model.infer_steps(u[:, 0:1, :], None)
    x2, y2 = model.infer_steps(u[:, 1:3, :], x1)
    x3, y3 = model.infer_steps(u[:, 3:15, :], x2)

    print(u.shape)
    print(y.shape)
    print(abs((y[:, 0:1, :] - y1).detach().cpu().numpy().round(-5)).sum())
    print(abs((y[:, 1:3, :] - y2).detach().cpu().numpy().round(-5)).sum())
    print(abs((y[:, 3:15, :] - y3).detach().cpu().numpy().round(-5)).sum())
