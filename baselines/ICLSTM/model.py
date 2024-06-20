from utils import *


class INCEPTION(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(INCEPTION, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.cnn2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.cnn2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.cnn3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.cnn3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 5), padding=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.cnn4 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        # x: (batch_size, 1, 28, 28) or (batch_size, 4, 28, 28)
        batch_size = x.size(0)
        y1 = self.cnn1(x)
        y2 = self.cnn2_2(self.cnn2_1(x))
        y3 = self.cnn3_2(self.cnn3_1(x))
        y4 = self.cnn4(self.pool4(x))
        y = torch.cat((y1, y2, y3, y4), dim=1)
        return y


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.fc1 = nn.Linear(28, 100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=200, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        # x: (batch_size, 28, 28)
        y = self.fc1(x)
        y, (hn, cn) = self.lstm(y)
        # use hn as output: (batch_size, hidden_size, 1, 1)
        return hn


class ICLSTM(nn.Module):
    def __init__(self):
        super(ICLSTM, self).__init__()
        self.inception1 = INCEPTION(1, 1)
        self.inception2 = INCEPTION(4, 1)
        self.lstm = LSTM()
        self.fc = nn.Linear(3336, NUM_LABELS)

    def forward(self, x):  # batch * 1500
        batch_size = x.size(0)
        y1 = self.inception1(x.view(batch_size, 1, 28, 28))
        y1 = self.inception2(y1.view(batch_size, 4, 28, 28)).view(batch_size, -1)
        y2 = self.lstm(x.view(batch_size, 28, 28)).view(batch_size, 200)
        y = self.fc(torch.cat((y1, y2), dim=1))
        return y


if __name__ == '__main__':
    xx = torch.randn(3, PKT_MAX_LEN)
    m = ICLSTM()
    out = m(xx)
    print(out.size())
