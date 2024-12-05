import torch

class ValueEstimator:
    def __init__(self, env, lr=1e-2):
        self.N = env.observation_space.shape[0]
        self.V = torch.nn.Sequential(
            torch.nn.Linear(self.N, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).double()

        with torch.no_grad():
            self.V[-1].weight.fill_(0)
            self.V[-1].bias.fill_(0)

        self.opt = torch.optim.Adam(self.V.parameters(), lr=lr)

    def predict(self, s_t):
        s_t = torch.tensor(s_t)
        return self.V(s_t).squeeze()

    def learn(self, V_pred, returns):
        returns = torch.tensor(returns)
        loss = torch.mean((V_pred - returns)**2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss