import torch
import torchvision.models as models
from torch import nn

from .utils import get_real_cat_features


class VAEImageOutputLayer(nn.Module):
    def __init__(self, params, data_parameters):
        super().__init__()
        self.image_shape = data_parameters["shape"]
        n_channels = self.image_shape[0]
        is_cifar = self.image_shape[-1] <= 32
        self.filters = params.filters
        self.n_classes = data_parameters["n_classes"]
        self.classification_layer = nn.Linear(params.latent_dim, self.n_classes)
        if is_cifar:
            self.projection_layer = nn.Linear(
                params.latent_dim,
                self.filters * self.image_shape[1] * self.image_shape[2],
            )
        else:
            self.projection_layer = nn.Linear(
                params.latent_dim,
                (self.filters * self.image_shape[1] * self.image_shape[2]) // 4,
            )

        encoder_layers = [
            BasicDCNNLayer(self.filters, self.filters),
            BasicDCNNLayer(self.filters, self.filters),
        ]
        if is_cifar:
            encoder_layers.append(
                nn.ConvTranspose2d(
                    self.filters,
                    2 * n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    dilation=1,
                )
            )
        else:
            encoder_layers.append(
                nn.ConvTranspose2d(
                    self.filters,
                    2 * n_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    output_padding=1,
                    bias=True,
                    dilation=1,
                )
            )
        self.model = nn.Sequential(*encoder_layers)
        self.is_cifar = is_cifar

    def forward(self, batch):
        projected = self.projection_layer(batch)
        if self.is_cifar:
            projected = projected.view(-1, self.filters, *self.image_shape[1:])
        else:
            projected = projected.view(
                -1, self.filters, self.image_shape[1] // 2, self.image_shape[2] // 2
            )
        images = self.model(projected)
        probs = self.classification_layer(batch)

        return (images, probs)


class VAEImageInputLayer(nn.Module):
    def __init__(self, params, data_parameters):
        super().__init__()
        self.image_shape = data_parameters["shape"]
        n_channels = self.image_shape[0] * 2
        self.filters = params.filters
        self.n_classes = data_parameters["n_classes"]
        is_cifar = self.image_shape[-1] <= 32
        self.embedding_layer = nn.Embedding(self.n_classes, params.embed_dim)
        encoder_layers = [
            BasicCNNLayer(n_channels, self.filters, init_layer=True, is_cifar=is_cifar),
            BasicCNNLayer(self.filters, self.filters),
            BasicCNNLayer(self.filters, self.filters),
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        output_shape = (
            self.encoder(torch.randn(1, n_channels, *self.image_shape[1:]))
            .view(1, -1)
            .shape[1]
        )
        output_shape += params.embed_dim
        self.final_layer = nn.Linear(output_shape, params.latent_dim * 2)

    def forward(self, batch):
        vals, masks = torch.chunk(batch, 2, dim=-1)
        images, labels, image_masks, label_masks = (
            vals[:, :-1],
            vals[:, -1],
            masks[:, :-1],
            masks[:, -1],
        )
        images = images.view(-1, *self.image_shape)
        image_masks = image_masks.view(-1, *self.image_shape)
        input_batch = torch.cat([images, image_masks], dim=1)
        output = self.encoder(input_batch).view(len(batch), -1)
        cat_output = self.embedding_layer(labels.long()) * label_masks[:, None]
        output = torch.cat([output, cat_output], dim=-1)

        return self.final_layer(output)


class ImagePolicyEncoder(nn.Module):
    def __init__(self, params, data_parameters):
        super().__init__()
        self.image_shape = data_parameters["shape"]
        n_channels = self.image_shape[0] * 2
        self.filters = params.filters
        self.aux_state = params.use_aux_state
        self.n_classes = data_parameters["n_classes"]
        is_cifar = self.image_shape[-1] <= 32
        if self.aux_state:
            n_channels += self.image_shape[0] * 3
        encoder_layers = [
            BasicCNNLayer(n_channels, self.filters, init_layer=True, is_cifar=is_cifar),
            BasicCNNLayer(self.filters, self.filters),
            BasicCNNLayer(self.filters, self.filters),
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        output_shape = (
            self.encoder(torch.randn(1, n_channels, *self.image_shape[1:]))
            .view(1, -1)
            .shape[1]
        )
        if self.aux_state:
            output_shape += self.n_classes
        self.final_layer = nn.Linear(output_shape, params.hidden_dim)

    def forward(self, batch):
        if self.aux_state:
            states, probs = batch[:, : -self.n_classes], batch[:, -self.n_classes :]
            vals, masks, uis, means, stds = torch.chunk(states, 5, dim=-1)
            uis = uis.view(-1, *self.image_shape)
            means = means.view(-1, *self.image_shape)
            stds = stds.view(-1, *self.image_shape)
        else:
            vals, masks = torch.chunk(batch, 2, dim=-1)
        vals = vals.view(-1, *self.image_shape)
        masks = masks.view(-1, *self.image_shape)
        input_batch = torch.cat(
            [vals, masks, uis, means, stds] if self.aux_state else [vals, masks], dim=1
        )
        output = self.encoder(input_batch).view(len(batch), -1)
        if self.aux_state:
            output = torch.cat([output, probs], dim=-1)
        return self.final_layer(output)


class ImageModels(nn.Module):
    def __init__(self, params, data_parameters):
        super().__init__()
        self.image_shape = data_parameters["shape"]
        n_channels = self.image_shape[0] * 2
        self.aux_state = params.use_aux_state
        self.n_classes = data_parameters["n_classes"]
        is_cifar = self.image_shape[-1] <= 32
        if self.aux_state:
            n_channels += self.image_shape[0] * 3
        if params.resnet == 18:
            self.resnet = ResNet18(n_channels, is_cifar=is_cifar)
        else:
            self.resnet = ResNet9(n_channels)
        output_shape = self.resnet(
            torch.randn(size=(1, n_channels, *self.image_shape[1:]))
        ).shape[1]
        if self.aux_state:
            output_shape += self.n_classes
        self.final_layer = nn.Linear(output_shape, data_parameters["n_classes"])

    def forward(self, batch):
        if self.aux_state:
            states, probs = batch[:, : -self.n_classes], batch[:, -self.n_classes :]
            vals, masks, uis, means, stds = torch.chunk(states, 5, dim=-1)
            uis = uis.view(-1, *self.image_shape)
            means = means.view(-1, *self.image_shape)
            stds = stds.view(-1, *self.image_shape)
        else:
            vals, masks = torch.chunk(batch, 2, dim=-1)
        vals = vals.view(-1, *self.image_shape)
        masks = masks.view(-1, *self.image_shape)
        input_batch = torch.cat(
            [vals, masks, uis, means, stds] if self.aux_state else [vals, masks], dim=1
        )
        output = self.resnet(input_batch)
        if self.aux_state:
            output = torch.cat([output, probs], dim=-1)
        return self.final_layer(output)


class BasicDCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += identity
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class BasicCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, init_layer=False, is_cifar=True):
        super().__init__()

        kernel_size, stride, padding = 3, 1, 1
        if not is_cifar and init_layer:
            kernel_size, stride, padding = 7, 2, 3
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_layer = init_layer
        if not self.init_layer:
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                dilation=1,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self.init_layer:
            identity = x
            out = self.conv2(out)
            out = self.bn2(out)

            out += identity
            out = self.relu(out)
        return out


class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""

    def __init__(self, block, layers, n_channels=3, is_cifar=True):
        super().__init__(block, layers)
        if is_cifar:
            self.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.bn1 = self._norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNet9(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet18(ResNetEncoder):
    def __init__(self, n_channels, is_cifar):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], n_channels, is_cifar)


class InputLayer(nn.Module):
    def __init__(self, mixed_input_layer):
        super().__init__()
        self.mixed_layer = mixed_input_layer
        self.n_features = len(mixed_input_layer.real_features) + len(
            mixed_input_layer.cat_features
        )

    def forward(self, input):
        batch, mask, future = (
            input[:, : self.n_features],
            input[:, self.n_features : self.n_features * 2],
            input[:, self.n_features * 2 :],
        )
        cat_mask = mask[:, self.mixed_layer.cat_features]  # B, C
        mixed_output = self.mixed_layer(batch)
        real_output = mixed_output.get("real_output", None)
        cat_output = mixed_output.get("cat_output", None)
        outputs = []
        if cat_output is not None:
            outputs.append(
                (cat_output * (1.0 - cat_mask.unsqueeze(2))).view(len(batch), -1)
            )
        if real_output is not None:
            outputs.append(real_output)
        outputs.append(mask)
        outputs.append(future)
        return torch.cat(outputs, dim=-1)


class VAEInputLayer(nn.Module):
    def __init__(self, mixed_input_layer):
        super().__init__()
        self.mixed_layer = mixed_input_layer

    def forward(self, input):
        (batch, mask) = torch.chunk(input, 2, dim=-1)
        cat_mask = mask[:, self.mixed_layer.cat_features]  # B, C
        mixed_output = self.mixed_layer(batch)
        real_output = mixed_output.get("real_output", None)
        cat_output = mixed_output.get("cat_output", None)
        outputs = []
        if cat_output is not None:
            outputs.append(
                (cat_output * (1.0 - cat_mask.unsqueeze(2))).view(len(batch), -1)
            )
        if real_output is not None:
            outputs.append(real_output)
        outputs.append(mask)
        return torch.cat(outputs, dim=-1)


class MixedInputLayer(nn.Module):
    def __init__(self, data_parameters, hidden_dim, embed_dim, add_label):
        super().__init__()
        (
            self.real_features,
            self.cat_features,
            self.cat_categories,
        ) = get_real_cat_features(data_parameters, add_label)
        if self.real_features:
            self.real_layer = nn.Linear(len(self.real_features), hidden_dim)
        if self.cat_features:
            self.cat_embed_layers = nn.ModuleList(
                [
                    nn.Embedding(self.cat_categories[idx], embed_dim)
                    for idx, _ in enumerate(self.cat_categories)
                ]
            )

    def forward(self, input):
        output = {}
        if self.real_features:
            real_input = input[:, self.real_features]
            output["real_output"] = self.real_layer(real_input)
        if self.cat_features:
            cat_input = input[:, self.cat_features].long()
            cat_output = []
            for idx in range(len(self.cat_features)):
                cat_output.append(
                    self.cat_embed_layers[idx](cat_input[:, idx]).unsqueeze(1)
                )  # B, 1, D
            output["cat_output"] = torch.cat(cat_output, dim=1)  # B, C, D
        return output


class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """

    def __init__(self, *args):
        super().__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)


class MemoryLayer(nn.Module):
    """
    If output=False, this layer stores its input in a static class dictionary
    `storage` with the key `id` and then passes the input to the next layer.
    If output=True, this layer takes stored tensor from a static storage.
    If add=True, it returns sum of the stored vector and an input,
    otherwise it returns their concatenation.
    If the tensor with specified `id` is not in `storage` when the layer
    with output=True is called, it would cause an exception.

    The layer is used to make skip-connections inside nn.Sequential network
    or between several nn.Sequential networks without unnecessary code
    complication.
    The usage pattern is
    ```
        net1 = nn.Sequential(
            MemoryLayer('#1'),
            MemoryLayer('#0.1'),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            MemoryLayer('#0.1', output=True, add=False),
            # here add cannot be True because the dimensions mismatch
            nn.Linear(768, 256),
            # the dimension after the concatenation with skip-connection
            # is 512 + 256 = 768
        )
        net2 = nn.Sequential(
            nn.Linear(512, 512),
            MemoryLayer('#1', output=True, add=True),
            ...
        )
        b = net1(a)
        d = net2(c)
        # net2 must be called after net1,
        # otherwise tensor '#1' will not be in `storage`
    ```
    """

    storage = {}

    def __init__(self, id, output=False, add=False):
        super().__init__()
        self.id = id
        self.output = output
        self.add = add

    def forward(self, input):
        if not self.output:
            self.storage[self.id] = input
            return input
        else:
            if self.id not in self.storage:
                err = "MemoryLayer: id '%s' is not initialized. "
                err += "You must execute MemoryLayer with the same id "
                err += "and output=False before this layer."
                raise ValueError(err)
            stored = self.storage[self.id]
            if not self.add:
                data = torch.cat([input, stored], -1)
            else:
                data = input + stored
            return data
