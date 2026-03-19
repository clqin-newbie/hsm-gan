import torch


class GeneratorLinear(torch.nn.Module):
    
    def __init__(self, z_dim=25, hidden_dims=[24, 48, 48, 22], condi=False):
        super().__init__()
        modules = []
        if condi:
            z_dim += 2
        hidden_dims.insert(0, z_dim)
        for i in range(len(hidden_dims)-1):
            modules.append(
                torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                torch.nn.InstanceNorm1d(hidden_dims[i+1]),
                torch.nn.ReLU())
            )
        self.generator = torch.nn.Sequential(*modules)
        
    def forward(self, z, c=None):
        if c is not None:
            z_c = torch.concat((z, c), dim=1)
        else:
            z_c = z
        out = self.generator(z_c)
        return torch.tanh(out)
        # return torch.sigmoid(out)
    

class DiscriminatorLinear(torch.nn.Module):
    def __init__(self, hidden_dims=[22, 48, 48, 24], condi=False):
        super().__init__()
        if condi:
            hidden_dims[0] += 2
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                torch.nn.InstanceNorm1d(hidden_dims[i+1]),
                torch.nn.ReLU())
            )
        modules.append(torch.nn.Linear(hidden_dims[-1], 1))

        self.discriminator = torch.nn.Sequential(*modules)

    def forward(self, x, y=None):
        if y is not None:
            x_y = torch.concat((x, y), dim=1)
        else:
            x_y = x
        out = self.discriminator(x_y)
        return out

