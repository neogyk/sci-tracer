import torch


def ucb1(reward, c, N, n):
    # trigger that points to the next node for visit
    _ucb1 = torch.mean(reward) + torch.mul(c, torch.sqrt(torch.log(N) / n))
    return _ucb1


class MCTreeSearch(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # https://arxiv.org/pdf/2103.04931
        return

    def _selection(self):
        return

    def _expansion(self):
        return

    def _simulation(self):
        # run the llm/neuralnetwork for the next state or solution generation
        return

    def _backpropagation(self):
        return

    def forward(self, tree: torch.Tensor):
        return
