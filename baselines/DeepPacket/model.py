from utils import *


# App: 1, 200, (1,4), (1,3)       1, 200, (200,5), (200,1)
# Tra: 1, 200, (1,5), (1,3)       1, 200, (200,4), (200,3)

class CNN_APP(nn.Module):
    __name__ = 'CNN_APPLICATION'

    def __init__(self):
        super(CNN_APP, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(1, 4), stride=(1, 3)),
            # nn.Dropout(0.05)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(200, 5), stride=(200, 1)),
            # nn.Dropout(0.05)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 495), stride=(1, 495))

        self.fc = nn.Sequential(
            nn.Linear(200, 150),
            nn.Dropout(0.05),
            nn.Linear(150, 100),
            nn.Dropout(0.05),
            nn.Linear(100, 70),
            nn.Dropout(0.05),
            nn.Linear(70, NUM_CLASS),
            nn.Dropout(0.05)
        )

    def forward(self, x):  # batch * 1500
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 1, -1)
        x = self.cnn1(x)    # batch * 200 * 1 * 499

        x = x.view(batch_size, 1, 200, -1)
        x = self.cnn2(x)    # batch * 200 * 1 * 495
        x = x.view(batch_size, 1, 200, -1)
        x = self.pool(x)

        x = x.view(batch_size, 200)
        out = self.fc(x)
        return out


class CNN_TRA(nn.Module):
    __name__ = 'CNN_TRAFFIC'

    def __init__(self):
        super(CNN_TRA, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(1, 5), stride=(1, 3)),
            # nn.Dropout(0.05)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(200, 4), stride=(200, 3)),
            # nn.Dropout(0.05)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 166), stride=(1, 166))

        self.fc = nn.Sequential(
            nn.Linear(200, 150),
            nn.Dropout(0.05),
            nn.Linear(150, 100),
            nn.Dropout(0.05),
            nn.Linear(100, 70),
            nn.Dropout(0.05),
            nn.Linear(70, NUM_CLASS),
            nn.Dropout(0.05)
        )

    def forward(self, x):  # batch * 1500
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 1, -1)
        x = self.cnn1(x)    # batch * 200 * 1 * 499

        x = x.view(batch_size, 1, 200, -1)
        x = self.cnn2(x)    # batch * 200 * 1 * 166
        x = x.view(batch_size, 1, 200, -1)
        x = self.pool(x)

        x = x.view(batch_size, 200)
        out = self.fc(x)
        return out


class SAE(nn.Module):
    __name__ = 'SAE'

    def __init__(self):
        super(SAE, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1500, 400),
            nn.Dropout(0.05),
            nn.Linear(400, 300),
            nn.Dropout(0.05),
            nn.Linear(300, 200),
            nn.Dropout(0.05),
            nn.Linear(200, 100),
            nn.Dropout(0.05),
            nn.Linear(100, 50),
            nn.Dropout(0.05),
            nn.Linear(50, NUM_CLASS),
            nn.Dropout(0.05)
        )

    def forward(self, x):  # batch * 1500
        out = self.fc(x)
        return out


class CNN_APP_D(nn.Module):
    __name__ = 'CNN_APPLICATION'

    def __init__(self):
        super(CNN_APP_D, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(1, 4), stride=(1, 3)),
            # nn.Dropout(0.05)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(200, 5), stride=(200, 1)),
            # nn.Dropout(0.05)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 495), stride=(1, 495))

        self.fc = nn.Sequential(
            nn.Linear(200, 150),
            # nn.Dropout(0.05),
            nn.Linear(150, 100),
            # nn.Dropout(0.05),
            nn.Linear(100, 70),
            # nn.Dropout(0.05),
            nn.Linear(70, NUM_CLASS),
            # nn.Dropout(0.05)
        )

    def forward(self, x):  # batch * 1500
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 1, -1)
        x = self.cnn1(x)    # batch * 200 * 1 * 499

        x = x.view(batch_size, 1, 200, -1)
        x = self.cnn2(x)    # batch * 200 * 1 * 495
        x = x.view(batch_size, 1, 200, -1)
        x = self.pool(x)

        x = x.view(batch_size, 200)
        out = self.fc(x)
        return out


class CNN_TRA_D(nn.Module):
    __name__ = 'CNN_TRAFFIC'

    def __init__(self):
        super(CNN_TRA_D, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(1, 5), stride=(1, 3)),
            # nn.Dropout(0.05)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(200, 4), stride=(200, 3)),
            # nn.Dropout(0.05)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 166), stride=(1, 166))

        self.fc = nn.Sequential(
            nn.Linear(200, 150),
            # nn.Dropout(0.05),
            nn.Linear(150, 100),
            # nn.Dropout(0.05),
            nn.Linear(100, 70),
            # nn.Dropout(0.05),
            nn.Linear(70, NUM_CLASS),
            # nn.Dropout(0.05)
        )

    def forward(self, x):  # batch * 1500
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 1, -1)
        x = self.cnn1(x)    # batch * 200 * 1 * 499

        x = x.view(batch_size, 1, 200, -1)
        x = self.cnn2(x)    # batch * 200 * 1 * 166
        x = x.view(batch_size, 1, 200, -1)
        x = self.pool(x)

        x = x.view(batch_size, 200)
        out = self.fc(x)
        return out


class SAE_D(nn.Module):
    __name__ = 'SAE'

    def __init__(self):
        super(SAE_D, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1500, 400),
            # nn.Dropout(0.05),
            nn.Linear(400, 300),
            # nn.Dropout(0.05),
            nn.Linear(300, 200),
            # nn.Dropout(0.05),
            nn.Linear(200, 100),
            # nn.Dropout(0.05),
            nn.Linear(100, 50),
            # nn.Dropout(0.05),
            nn.Linear(50, NUM_CLASS),
            # nn.Dropout(0.05)
        )

    def forward(self, x):  # batch * 1500
        out = self.fc(x)
        return out