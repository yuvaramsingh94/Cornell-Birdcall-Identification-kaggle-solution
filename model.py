import torch
import torch.nn as nn


class Autopool(nn.Module):
    def __init__(
        self,
        input_size,
    ):
        super(Autopool, self).__init__()
        self.alpha = nn.Parameter(requires_grad=True)
        self.alpha.data = torch.ones(
            [input_size], dtype=torch.float32, requires_grad=True, device=device
        )
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        sigmoid_output = self.sigmoid_layer(x)
        alpa_mult_out = torch.mul(sigmoid_output, self.alpha)
        max_tensor = torch.max(alpa_mult_out, dim=1)
        max_tensor_unsqueezed = max_tensor.values.unsqueeze(dim=1)
        softmax_numerator = torch.exp(alpa_mult_out.sub(max_tensor_unsqueezed))
        softmax_den = torch.sum(softmax_numerator, dim=1)
        softmax_den = softmax_den.unsqueeze(dim=1)
        weights = softmax_numerator / softmax_den
        final_out = torch.sum(torch.mul(sigmoid_output, weights), dim=1)
        return final_out, sigmoid_output


class Yuvsub(nn.Module):
    def __init__(self, args):
        super(Yuvsub, self).__init__()
        self.species = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, args["params"]["n_classes"]),
        )

    def forward(self, GAP):

        spe = self.species(GAP)

        return spe


class Yuvgru(nn.Module):
    def __init__(self, args):
        super(Yuvgru, self).__init__()
        self.gru_layer = torch.nn.GRU(
            input_size=1024,
            hidden_size=args["gru_hidden_size"],
            num_layers=args["gru_layers"],
            dropout=0,
            bidirectional=args["gru_bidirectional"],
        )

    def forward(self, x):
        Routput, hn = self.gru_layer(
            x
        )  ## not passing the hiddenstate and it will be default to 0
        return Routput, hn


class YuvNet(nn.Module):
    def __init__(self, args):
        super(YuvNet, self).__init__()

        self.model = models.resnet18(pretrained=True)
        del self.model.fc
        self.model.fc = Yuvsub(args)
        # self.Gru = Yuvgru(args)

        self.autopool = Autopool(args["params"]["n_classes"])

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, H, W)

        # print('c_in shape ',c_in.shape)
        # print('c_in type ',c_in.dtype)
        spe = self.model(c_in)
        # print('shape of spe ',spe.shape)
        spe = spe.view(batch_size, time_steps, -1)
        final_output, sigmoid_output = self.autopool(spe)

        # return final_output, sigmoid_output
        return final_output, sigmoid_output
