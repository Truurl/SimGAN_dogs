from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import config as cfg

class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=64):
        super(ResnetBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, 1, 1),
            # nn.BatchNorm2d(nb_features),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1),
            # nn.BatchNorm2d(nb_features)
            # nn.LeakyReLU(),
            # nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        convs = self.convs(x)
        # sum = convs + x
        output = self.relu(convs + x)
        return output


class Refiner(nn.Module):
    def __init__(self, block_num, in_features, nb_features=64, num_heads=4):
        super(Refiner, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_features, nb_features, 3, stride=1, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(nb_features)
        )

        blocks = []
        for i in range(block_num):
            blocks.append(ResnetBlock(nb_features, nb_features))

        self.resnet_blocks = nn.Sequential(*blocks)

        if cfg.attention:
            self.attention = nn.MultiheadAttention(embed_dim=nb_features, num_heads=num_heads)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(nb_features, in_features, 1, 1, 0),
            # nn.Tanh()
        )

    def forward(self, x):
        conv_1 = self.conv_1(x)

        res_block = self.resnet_blocks(conv_1)

        if cfg.attention:
            batch_size, channels, height, width = res_block.shape
            query = res_block.view(batch_size, channels, -1).permute(0, 2, 1)
            # Apply multi-head self-attention
            att_output, _ = self.attention(query, query, query)

            # Reshape the output back to the original shape
            att_output = att_output.permute(0, 2, 1).view(batch_size, channels, height, width)

            output = self.conv_2(att_output)
        else:
            output = self.conv_2(res_block)

        return output.clone()


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(96),

            nn.Conv2d(96, 64, 3, 2, 1),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(64),

            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(32),

            nn.Conv2d(32, 2, 1, 1, 0),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(2),

            # IF USES BCEWithLogits DON"T UNCOMMENT
            # nn.Softmax()
        )

    def forward(self, x):
        convs = self.convs(x)
        # print(convs.size())
        output = convs.reshape(convs.size(0), -1, 2)
        return output

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()


        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
