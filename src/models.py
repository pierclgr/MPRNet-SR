import torch
from torch import nn
from src.exceptions import InvalidScaleException
import torch.nn.functional as F


class Conv2d1x1(nn.Module):
    def __init__(self, input_channels: int, reduction_factor: int = 1, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = input_channels // reduction_factor

        self.out_channels = out_channels

        # define the 1x1 convolution layer
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2d(nn.Module):
    def __init__(self, input_channels: int, kernel_size: int):
        super().__init__()

        padding_size = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                              kernel_size=(kernel_size, kernel_size), groups=input_channels,
                              padding=(padding_size, padding_size))

    def forward(self, input_tensor):
        return self.conv(input_tensor)


class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels: int, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = input_channels

        self.conv = Conv2d1x1(input_channels=input_channels, out_channels=out_channels)

    def forward(self, input_tensor):
        return self.conv(input_tensor)


class TwoFoldAttentionModule(nn.Module):
    class ChannelUnit(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            self.in_channels = input_channels

            # we define a global average pooling layer that extracts first-order statistics of features
            self.global_avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)

            # we then define two 1x1 convolutions that will work on half of the input channels and that will produce
            # half of the output channels; these two 1x1 convolutions will not reduce the number of channels of the
            # input tensor
            conv_1x1_input_channels = input_channels // 2
            self.conv1x1_1 = Conv2d1x1(input_channels=conv_1x1_input_channels)
            self.conv1x1_2 = Conv2d1x1(input_channels=conv_1x1_input_channels)

        def forward(self, input_tensor):
            # first, we feed the input to the global average pooling layer to extract first-order statistics of features
            # input_size = (N, in_channels, H, W)
            first_order_statistics = self.global_avg_pooling(input_tensor)  # output_size = (N, in_channels, 1, 1)

            # after producing first order statistic of features, we need to split the output tensor of the global
            # average pooling into two tensors along the channel dimension
            half_channels = self.in_channels // 2
            first_half_input, second_half_input = torch.split(first_order_statistics,
                                                              split_size_or_sections=half_channels, dim=1)
            # output: two tensors of size (N, in_channels/2, 1, 1)

            # now that we obtained the two halves of the channels, we feed them respectively to the first and second 1x1
            # convolutions that will produce half the output channels
            first_half_output = self.conv1x1_1(first_half_input)  # output_size = (N, in_channels/2, 1, 1)
            second_half_output = self.conv1x1_2(second_half_input)  # output_size = (N, in_channels/2, 1, 1)

            # we then concatenate the two halves of the output channels to get all the output channels
            concatenated_halves = torch.cat((first_half_output, second_half_output), dim=1)
            # output_size = (N, in_channels, 1, 1)

            # now, we compute element-wise multiplication of the input tensor (x) and the output tensor produced by the
            # concatenation operation
            output = torch.mul(concatenated_halves, input_tensor)  # output_size = (N, in_channels, 1, 1)

            return output

    class PositionalUnit(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            # define the average pooling and the max pooling layers with a large kernel
            self.avg_pooling = nn.AvgPool2d(kernel_size=(7, 7))
            self.max_pooling = nn.MaxPool2d(kernel_size=(7, 7))

            # define the final convolutional layer
            kernel_size = 7
            padding_size = kernel_size // 2
            self.conv2d_1 = nn.Conv2d(in_channels=input_channels * 2, out_channels=input_channels,
                                      kernel_size=(kernel_size, kernel_size),
                                      padding=(padding_size, padding_size))

        def forward(self, input_tensor):
            # get the spatial dimensions of the input tensor
            height = input_tensor.size()[2]
            width = input_tensor.size()[3]

            # first, we feed the input tensor to the average pooling layer and to the max pooling layer
            output_max_pool = self.max_pooling(input_tensor)
            output_avg_pool = self.avg_pooling(input_tensor)

            # then, we concatenate the two outputs produced by the max
            output_pool = torch.cat((output_max_pool, output_avg_pool), dim=1)

            # now, we upsample the output concatenation to recover spatial dimensions
            upsampled_out = F.interpolate(output_pool, size=(height, width), mode="bilinear", align_corners=False)

            # once we upsampled the concatenation, we apply
            output = self.conv2d_1(upsampled_out)

            return output

    def __init__(self, input_channels: int):
        super().__init__()

        # define first 1x1 convolution layer
        self.conv1x1_1 = Conv2d1x1(input_channels=input_channels, reduction_factor=16)

        # now, we define respectively the Channel Unit (CA Unit) and the Positional Unit (Pos Unit)
        self.ca_unit = self.ChannelUnit(input_channels=self.conv1x1_1.out_channels)
        self.pos_unit = self.PositionalUnit(input_channels=self.conv1x1_1.out_channels)

        # define the last 1x1 convolution layer used to recover original channel dimension
        self.conv1x1_2 = Conv2d1x1(input_channels=self.conv1x1_1.out_channels, out_channels=input_channels)

        # define the sigmoid function to generate the final attention mast
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        # first, we pass the input tensor to the first 1x1 convolution layer
        conv_1x1_1_out = self.conv1x1_1(input_tensor)

        # then, we pass the output produced by the 1x1 convolution layer to the Channel unit
        ca_unit_out = self.ca_unit(conv_1x1_1_out)

        # we pass the output produced by the 1x1 convolution layer to the Positional Unit
        pos_unit_out = self.pos_unit(conv_1x1_1_out)

        # we compute element-wise sum of the two tensors produced by the two units
        sum_output = torch.add(ca_unit_out, pos_unit_out)

        # we feed the aggregated tensors to the last 1x1 conv layer to recover channel dimensions
        conv_1x1_2_out = self.conv1x1_2(sum_output)

        # we feed the output of the 1x1 conv layer to the sigmoid layer to compute the final attention mast
        sigmoid_out = self.sigmoid(conv_1x1_2_out)

        # finally, we compute the element-wise multiplication between the computed final attention mask and the input
        # tensor
        output = torch.mul(input_tensor, sigmoid_out)

        return output


class AdaptiveResidualBlock(nn.Module):
    class BottleneckPath(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            # define the first depthwise convolutional layer with kernel size 3x3
            self.dw_conv_1 = DepthwiseConv2d(input_channels=input_channels, kernel_size=3)

            # define the first pointwise convolutional layer, followed by LeakyReLU
            self.pw_conv_1 = PointwiseConv2d(input_channels=input_channels)
            self.lrelu_1 = nn.LeakyReLU()

            # define the second depthwise convolutional layer with kernel size 3x3
            self.dw_conv_2 = DepthwiseConv2d(input_channels=input_channels, kernel_size=3)

            # define the TFAM layer, followed by LeakyReLU
            self.tfam = TwoFoldAttentionModule(input_channels=input_channels)
            self.lrelu_2 = nn.LeakyReLU()

            # define the second pointwise convolution layer
            self.pw_conv_2 = PointwiseConv2d(input_channels=input_channels)

        def forward(self, input_tensor):
            # first, we feed the input tensor to the first depthwise convolutional layer
            dw_conv_1_out = self.dw_conv_1(input_tensor)

            # then, we feed the output to the first pointwise convolutional layer and to the first LReLU layer
            pw_conv_1_out = self.pw_conv_1(dw_conv_1_out)
            lrelu_1_out = self.lrelu_1(pw_conv_1_out)

            # after this, we feed the output to the second depthwise convolutional layer
            dw_conv_2_out = self.dw_conv_2(lrelu_1_out)

            # then, we feed the output to the TFAM module and to the second LReLU
            tfam_out = self.tfam(dw_conv_2_out)
            lrelu_2_out = self.lrelu_2(tfam_out)

            # finally, the output is fed to the second pointwise convolutional layer and returned
            output = self.pw_conv_2(lrelu_2_out)

            return output

    class AdaptivePath(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            # define the global average pooling layer
            self.global_avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)

            # define the pointwise convolution layer
            self.pw_conv = PointwiseConv2d(input_channels=input_channels)

        def forward(self, input_tensor):
            # first, we feed the input tensor to the global average pooling layer
            global_avg_out = self.global_avg_pooling(input_tensor)

            # finally, we feed the output to the pointwise convolution layer
            output = self.pw_conv(global_avg_out)

            return output

    class ResidualPath(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            # define the depthwise convolution layer with kernel size 3x3
            self.dw_conv = DepthwiseConv2d(input_channels=input_channels, kernel_size=3)

        def forward(self, input_tensor):
            return self.dw_conv(input_tensor)

    def __init__(self, input_channels: int):
        super().__init__()

        # first, we define the bottleneck path
        self.bn_path = self.BottleneckPath(input_channels=input_channels)

        # second, we define the adaptive path
        self.ad_path = self.AdaptivePath(input_channels=input_channels)

        # third, we define the residual path
        self.res_path = self.ResidualPath(input_channels=input_channels)

    def forward(self, input_tensor):
        # as first step, we pass the input tensor to the bottleneck path
        bn_path_out = self.bn_path(input_tensor)

        # then, we compute element-wise sum between the input tensor and the output of the bottleneck path
        sum_bn_input = torch.add(input_tensor, bn_path_out)

        # we feed the sum to the residual path
        res_path_out = self.res_path(sum_bn_input)

        # then, we pass the input tensor to the adaptive path
        ad_path_out = self.ad_path(input_tensor)

        # finally, we compute the output as the element-wise sum between the output of the residual path and the output
        # of the adaptive path
        output = torch.add(res_path_out, ad_path_out)

        return output


class ResidualConcatenationBlock(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()

        # the definition of a Residual Concatenation Block contains three different Adaptive Residual Blocks that share
        # the same weights; in order to implement this, we simply define one RCB that will be called three times in the
        # forward function
        self.arb = AdaptiveResidualBlock(input_channels=input_channels)

        # define the first 1x1 convolutional layer
        first_conv_input_channels = input_channels * 2
        self.conv_1x1_1 = Conv2d1x1(input_channels=first_conv_input_channels, out_channels=input_channels)

        # define the second 1x1 convolutional layer
        second_conv_input_channels = input_channels * 3
        self.conv_1x1_2 = Conv2d1x1(input_channels=second_conv_input_channels, out_channels=input_channels)

        # define the third 1x1 convolutional layer
        third_conv_input_channels = input_channels * 4
        self.conv_1x1_3 = Conv2d1x1(input_channels=third_conv_input_channels, out_channels=input_channels)

        # # after implementing the three layers, we first delete their weights
        # del self.conv_1x1_1.weight
        # del self.conv_1x1_2.weight
        # del self.conv_1x1_3.weight
        #
        # # then, we create shared weights to assign to the three 1x1 conv layers
        # self.shared_weights = nn.Parameter(torch.randn(input_channels, input_channels * 2))
        # self.conv_1x1_2_base_weights = nn.Parameter(torch.randn(input_channels, input_channels * 3))
        # self.conv_1x1_3_base_weights = nn.Parameter(torch.randn(input_channels, input_channels * 4))

    def forward(self, input_tensor):
        # first, feed the input tensor to the ARB
        arb_1_out = self.arb(input_tensor)

        # second, we concatenate the output of the first ARB block with the input tensor
        concat_1_out = torch.cat((input_tensor, arb_1_out), dim=1)

        # after this, we feed the concatenation to the first 1x1 convolutional layer
        conv_1x1_1_out = self.conv_1x1_1(concat_1_out)

        # we feed the output of the 1x1 convolutional layer to the ARB
        arb_2_out = self.arb(conv_1x1_1_out)

        # we concatenate the output of the second ARB block with the previous concatenation
        concat_2_out = torch.cat((concat_1_out, arb_2_out), dim=1)

        # we feed the concatenation to the second 1x1 conv
        conv_1x1_2_out = self.conv_1x1_2(concat_2_out)

        # we feed the output of the second 1x1 conv layer to the ARB
        arb_3_out = self.arb(conv_1x1_2_out)

        # we concatenate the output of the third ARB block with the previous concatenation
        concat_3_out = torch.cat((concat_2_out, arb_3_out), dim=1)

        # finally, we feed the concatenation to the third  and last 1x1 conv layer
        output = self.conv_1x1_3(concat_3_out)

        return output


class ResidualModule(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()

        # define the first Residual Concatenation Block
        self.rcb_1 = ResidualConcatenationBlock(input_channels=input_channels)

        # define the first 1x1 convolutional layer
        first_conv_input_channels = input_channels * 2
        self.conv_1x1_1 = Conv2d1x1(input_channels=first_conv_input_channels, out_channels=input_channels)

        # define the second Residual Concatenation Block
        self.rcb_2 = ResidualConcatenationBlock(input_channels=input_channels)

        # define the second 1x1 convolutional layer
        second_conv_input_channels = input_channels * 3
        self.conv_1x1_2 = Conv2d1x1(input_channels=second_conv_input_channels, out_channels=input_channels)

        # define the third Residual Concatenation Block
        self.rcb_3 = ResidualConcatenationBlock(input_channels=input_channels)

        # define the third 1x1 convolutional layer
        third_conv_input_channels = input_channels * 4
        self.conv_1x1_3 = Conv2d1x1(input_channels=third_conv_input_channels, out_channels=input_channels)

    def forward(self, h_sfe):
        # first, feed the input tensor (h_sfe) to the first RCB block
        rcb_1_out = self.rcb_1(h_sfe)

        # second, we concatenate the output of the first RCB block with the input tensor (h_sfe)
        concat_1_out = torch.cat((h_sfe, rcb_1_out), dim=1)

        # after this, we feed the concatenation to the first 1x1 convolutional layer
        conv_1x1_1_out = self.conv_1x1_1(concat_1_out)

        # we feed the output of the 1x1 convolutional layer to the second RCB
        rcb_2_out = self.rcb_2(conv_1x1_1_out)

        # we concatenate the output of the second RCB block with the previous concatenation
        concat_2_out = torch.cat((concat_1_out, rcb_2_out), dim=1)

        # we feed the concatenation to the second 1x1 conv
        conv_1x1_2_out = self.conv_1x1_2(concat_2_out)

        # we feed the output of the second 1x1 conv layer to the third RCB
        rcb_3_out = self.rcb_3(conv_1x1_2_out)

        # we concatenate the output of the third ARB block with the previous concatenation
        concat_3_out = torch.cat((concat_2_out, rcb_3_out), dim=1)

        # finally, we feed the concatenation to the third  and last 1x1 conv layer
        h_rm = self.conv_1x1_3(concat_3_out)

        return h_rm


class FeatureModule(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()

        # define the first layer, which is a TFAM
        self.tfam = TwoFoldAttentionModule(input_channels=input_channels)

        # define the second layer, which is a 3x3 conv layer
        kernel_size = 3
        padding_size = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                              kernel_size=(kernel_size, kernel_size), padding=padding_size)

    def forward(self, h_rm, h_sfe):
        # first, we feed the input tensor (h_rm) to the tfam layer
        tfam_out = self.tfam(h_rm)

        # then, we feed the output of the tfam layer to the convolutional layer
        h_gfe = self.conv(tfam_out)

        # finally, we compute the element-wise sum between the output of the convolutional layer and the shallow
        # features
        h_fm = torch.add(h_gfe, h_sfe)

        return h_fm


class UpNetModule(nn.Module):
    class Upsample2x(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            kernel_size = 3
            padding_size = kernel_size // 2

            # define the submodule that produces a feature map upsampled by 2x
            self.conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 4, kernel_size=(3, 3),
                                  padding=padding_size)
            self.pix_shuf = nn.PixelShuffle(upscale_factor=2)

        def forward(self, input_tensor):
            # feed the input tensor to the conv layer
            conv_out = self.conv(input_tensor)

            # feed the output of the conv layer to the pixel shuffle layer
            pix_shuf_out = self.pix_shuf(conv_out)

            return pix_shuf_out

    class Upsample3x(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            kernel_size = 3
            padding_size = kernel_size // 2

            # define the submodule that produces a feature map upsampled by 3x
            self.conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 9, kernel_size=(3, 3),
                                  padding=padding_size)
            self.pix_shuf = nn.PixelShuffle(upscale_factor=3)

        def forward(self, input_tensor):
            # feed the input tensor to the conv layer
            conv_out = self.conv(input_tensor)

            # feed the output of the conv layer to the pixel shuffle layer
            pix_shuf_out = self.pix_shuf(conv_out)

            return pix_shuf_out

    class Upsample4x(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()

            # define the first submodule that produces a feature map upsampled by 4x
            self.upsample_4x = nn.Sequential(UpNetModule.Upsample2x(input_channels=input_channels),
                                             UpNetModule.Upsample2x(input_channels=input_channels))

        def forward(self, input_tensor):
            # feed the input tensor to the upsampler
            return self.upsample_4x(input_tensor)

    def __init__(self, input_channels: int):
        super().__init__()

        # define the submodule that produces a feature map upsampled by 2x
        self.upsample_2x = self.Upsample2x(input_channels=input_channels)

        # define the submodule that produces a feature map upsampled by 3x
        self.upsample_3x = self.Upsample3x(input_channels=input_channels)

        # define the submodule that produces a feature map upsampled by 3x
        self.upsample_4x = self.Upsample4x(input_channels=input_channels)

    def forward(self, h_fm, scale: int):
        # feed the input tensor to one of the upsamplers according to the given scale
        if scale == 2:
            upsampled = self.upsample_2x(h_fm)
        elif scale == 3:
            upsampled = self.upsample_3x(h_fm)
        elif scale == 4:
            upsampled = self.upsample_4x(h_fm)
        else:
            raise InvalidScaleException(f"Scale factor {scale} is invalid, select between 2, 3 or 4")

        return upsampled


class MultiPathResidualNetwork(nn.Module):
    def __init__(self, input_channels: int, n_features: int = 64):
        super().__init__()

        # initialize initial shallow feature extractor
        kernel_size = 3
        padding_size = kernel_size // 2
        self.sfe = nn.Conv2d(in_channels=input_channels, out_channels=n_features, kernel_size=(3, 3),
                             padding=padding_size)

        # define the Residual Module
        self.rm = ResidualModule(input_channels=n_features)

        # define the Feature Module
        self.fm = FeatureModule(input_channels=n_features)

        # define teh UpNet Module
        self.upnet = UpNetModule(input_channels=n_features)

        # define the final 3x3 convolution that restores the channels to three RGB channels
        self.out_conv = nn.Conv2d(in_channels=n_features, out_channels=input_channels, kernel_size=(3, 3),
                                  padding=padding_size)

    def forward(self, lrs, scale: int):
        # input is the batch of low resolution images, with shape (N, 3, 64, 64)
        h_sfe = self.sfe(lrs)  # output size (N, 64, 64, 64)

        # feed h_sfe to the residual module
        h_rm = self.rm(h_sfe)  # output size (N, 64, 64, 64)

        # feed h_rm and h_sfe to the feature module
        h_fm = self.fm(h_rm, h_sfe)  # output size (N, 64, 64, 64)

        # feed h_fm to the upnet module
        upscaled_fm = self.upnet(h_fm, scale)  # output size (N, 64, 64 * scale, 64 * scale)

        # feed upscaled feature map to the last 3x3 conv layer to get the final hr image in 3 RGB channels
        srs = self.out_conv(upscaled_fm)  # output size (N, 3,  64 * scale, 64 * scale)

        return srs
