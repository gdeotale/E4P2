from torch import nn
from torch.autograd import Variable

ZDIMS = 256
dropout_value = 0.1
leaky_value = 0.1
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #128x128x3
        # ENCODER
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_value)
        )
        #64x64x64
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leaky_value)
        )
        #32x32x128
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_value)
        ) 
        #16x16x256
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leaky_value),
        ) 
        #8x8x256
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 
        #1x1x256
        self.fc21 = nn.Linear(512, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(512, ZDIMS)  # logvariance layer
        
        # DECODER
        self.fc = nn.Sequential(
            nn.Linear(ZDIMS, 8*8*512),
            nn.LeakyReLU(leaky_value),
        )
        self.convd1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leaky_value)
        )
            # x2
        self.trans1 = nn.Sequential(    
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_value)
        )
            # x2
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leaky_value)
        )
            # x2
        self.trans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leaky_value)
        )
            # x2
        self.trans4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def encode(self, x: Variable) -> (Variable, Variable):
        # h1 is [128, 400]
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)  # type: Variable
        h1 = x.view(-1, 512)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z: Variable) -> Variable:
        z = z.view(z.size(0), -1)
        y_ = self.fc(z)
        y_ = y_.view(y_.size(0), 512, 8, 8)
        x = self.convd1(y_)
        x = self.trans1(x)
        x = self.trans2(x)
        x = self.trans3(x)
        x = self.trans4(x)
        return x

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar