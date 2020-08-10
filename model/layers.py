from torch import nn

Pool = nn.MaxPool2d

# class Merge(nn.Module):
#     def __init__(self, x_dim, y_dim):
#         super(Merge, self).__init__()
#         self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)
#
#     def forward(self, x):
#         return self.conv(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        #self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = None

    def forward(self, x):
        #assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip_convo = False
        else:
            self.need_skip_convo = True
        
    def forward(self, x):
        #adjust output dim
        if self.need_skip_convo:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        #element-wise addition, feature fusion
        out += residual
        return out 

class Hourglass(nn.Module):

    #n is the oreder of a single hourglass module, f is number of feature map
    def __init__(self, n, f, bn=None):
        super(Hourglass, self).__init__()
        self.skip_con = Residual(f, f)

        # Lower branch
        self.pool = Pool(2, 2)
        self.bottom_up_res = Residual(f, f)

        # Recursive hourglass
        self.n = n
        if self.n > 1:
            self.recursive = Hourglass(n-1, f, bn=bn)
        else:
            self.recursive = Residual(f, f)

        self.top_down_res = Residual(f, f)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        #upper connection
        skip_f  = self.skip_con(x)

        #lower connection
        pool = self.pool(x)
        bottom_up_res = self.bottom_up_res(pool)
        recursive = self.recursive(bottom_up_res)   #In hg, res includs: 64, 32, 16, 8 if input_res=256
        top_down_res = self.top_down_res(recursive)
        up_sample  = self.up_sample(top_down_res)
        return skip_f + up_sample
