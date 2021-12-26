import norse
from norse.torch.functional.stdp import STDPParameters, STDPState
import norse.torch.functional.stdp
import torch


def super_threshold(x:torch.Tensor, alpha:torch.Tensor):
    return torch.gt(x, torch.as_tensor(0.0)).to(x.dtype)

class LIFCalc():
    def __init__(self, p:norse.torch.LIFParameters) -> None:
        self.p = p
        self.cell = norse.torch.LIFCell(self.p)
        self.state = norse.torch.LIFFeedForwardState(
            v=torch.as_tensor(0),
            i=torch.as_tensor(0)
        )
        size = torch.Size([1])
        self.state = norse.torch.LIFFeedForwardState(
            v=torch.full(
                size=size,
                fill_value=self.p.v_leak.detach()
            ),
            i=torch.zeros(size=size)
        )

    def tick(self, x):
        out, state = self.cell(x, state=self.state)
        assert isinstance(state, norse.torch.LIFFeedForwardState)

        out_v, state_v = self._calc(i_input=x, state=self.state)
        torch.testing.assert_allclose(out, out_v)
        torch.testing.assert_allclose(state.v, state_v.v)
        torch.testing.assert_allclose(state.i, state_v.i)

        self.state = state
        return out

    def _calc(self, i_input, state):
        dt = 0.001
        p = self.p

        dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
        di = -dt * p.tau_syn_inv * state.i

        # 1 if state.v + dv > p.v_th else 0
        out_new = super_threshold(state.v + dv - p.v_th, p.alpha)

        v_new = (1-out_new) * (state.v+dv) + out_new * p.v_reset
        i_new = state.i + di + i_input

        return out_new, norse.torch.LIFFeedForwardState(v=v_new, i=i_new)

class STDPCalc():
    def __init__(self, p:STDPParameters) -> None:
        self.p = p

    def tick(self):
        pass

    def _calc(self):
        pass

def test_stdp():
    dt=0.001
    n_batches = 4
    n_pre = 1
    n_post = 1
    W = torch.ones(size=(n_post, n_pre)) * 0.1
    state = STDPState(
        t_pre=torch.zeros(size=(n_batches, n_pre)),
        t_post=torch.zeros(size=(n_batches, n_post))
    )
    z_pre = torch.zeros((n_batches, 1, 1))
    z_pre[1,0,0] = 1
    z_pre[2,0,0] = 1
    z_post = torch.zeros((n_batches, 1, 1))
    z_post[1,0,0] = 1

    p = STDPParameters(
        eta_minus=1e-1,
        eta_plus=3e-1,
        stdp_algorithm="additive",
        mu=0, # exponent for multiplicative std_algorithm
        hardbound=True, # Clip
        convolutional=False
    )
    print("===== W Before =====")
    print(W)
    w0 = W
    for i in range(n_batches):
        # Each iteration state is decayed
        # then the decayed state is used to update W
        # https://github.com/norse/norse/blob/7aa37658d8dbde31563c6b8a6821e1e6421ac545/norse/torch/functional/stdp.py#L129
        W, state = norse.torch.functional.stdp.stdp_step_linear(
            z_pre=z_pre[i],
            z_post=z_post[i],
            w=W,
            state_stdp=state,
            p_stdp=p,
            dt=dt
        )
        # Neither spike - no change
        # Both spike - W increases
        # pre spikes post doesn't spike - W decreases
        # Neither spike - no change
        print("===== W after =====")
        print(W)


def run_loop():
    p = norse.torch.LIFParameters(
        tau_syn_inv=torch.as_tensor(200.),
        tau_mem_inv=torch.as_tensor(100.),
        v_leak=torch.as_tensor(0.),
        v_th=torch.as_tensor(2.),
        v_reset=torch.as_tensor(0.),
        method='super',
        alpha=torch.as_tensor(100.)
    )
    lif_calc = LIFCalc(p)
    for _ in range(20):
        x = torch.ones(1)
        out = lif_calc.tick(x)
        print(out, lif_calc.state.v, lif_calc.state.i)

def main():
    test_stdp()

if __name__ == "__main__":
    main()

