import math

import torch
import torch.nn as nn

class AvgPool(nn.Module):
    def __init__(self, num_spixels):
        super(AvgPool, self).__init__()
        if isinstance(num_spixels, tuple):
            assert len(num_spixels) == 2
        elif num_spixels is not None:
            x = int(math.sqrt(num_spixels))
            assert x * x == num_spixels
            num_spixels = (x, x)
        self.num_spixels = num_spixels
        self.module = nn.AdaptiveAvgPool2d(num_spixels)

    def forward(self, x):
        batch_size, dim, _, _ = x.shape
        x = self.module(x)
        return x.reshape(batch_size, dim, -1).transpose(-1, -2), None

class TokenClusteringBlock(nn.Module):
    def __init__(self, num_spixels=None, n_iters=5, temperture=1., window_size=7):
        super().__init__()
        if isinstance(num_spixels, tuple):
            assert len(num_spixels) == 2
        elif num_spixels is not None:
            x = int(math.sqrt(num_spixels))
            assert x * x == num_spixels
            num_spixels = (x, x)
        self.num_spixels = num_spixels
        self.n_iters = n_iters
        self.temperture = temperture
        assert window_size % 2 == 1
        self.r = window_size // 2

    def calc_init_centroid(self, images, num_spixels_width, num_spixels_height):
        """
        calculate initial superpixels

        Args:
            images: torch.Tensor
                A Tensor of shape (B, C, H, W)
            spixels_width: int
                initial superpixel width
            spixels_height: int
                initial superpixel height

        Return:
            centroids: torch.Tensor
                A Tensor of shape (B, C, H * W)
            init_label_map: torch.Tensor
                A Tensor of shape (B, H * W)
            num_spixels_width: int
                A number of superpixels in each column
            num_spixels_height: int
                A number of superpixels int each raw
        """
        batchsize, channels, height, width = images.shape
        device = images.device

        centroids = torch.nn.functional.adaptive_avg_pool2d(
            images, (num_spixels_height, num_spixels_width)
        )

        with torch.no_grad():
            num_spixels = num_spixels_width * num_spixels_height
            labels = (
                torch.arange(num_spixels, device=device)
                .reshape(1, 1, *centroids.shape[-2:])
                .type_as(centroids)
            )
            init_label_map = torch.nn.functional.interpolate(
                labels, size=(height, width), mode="nearest"
            ).type_as(centroids)
            init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

        init_label_map = init_label_map.reshape(batchsize, -1)
        centroids = centroids.reshape(batchsize, channels, -1)

        return centroids, init_label_map

    def forward(self, pixel_features, num_spixels=None):
        if num_spixels is None:
            num_spixels = self.num_spixels
            assert num_spixels is not None
        else:
            if isinstance(num_spixels, tuple):
                assert len(num_spixels) == 2
            else:
                x = int(math.sqrt(num_spixels))
                assert x * x == num_spixels
                num_spixels = (x, x)
        num_spixels_height, num_spixels_width = num_spixels
        num_spixels = num_spixels_width * num_spixels_height
        spixel_features, init_label_map = self.calc_init_centroid(
            pixel_features, num_spixels_width, num_spixels_height
        )

        device = init_label_map.device
        spixels_number = torch.arange(num_spixels, device=device)[None, :, None]
        relative_labels_widths = init_label_map[:, None] % num_spixels_width - spixels_number % num_spixels_width
        relative_labels_heights = init_label_map[:, None] // num_spixels_width - spixels_number // num_spixels_width
        mask = torch.logical_and(torch.abs(relative_labels_widths) <= self.r, torch.abs(relative_labels_heights) <= self.r)
        mask_dist = (~mask) * 1e16

        pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)  # (B, C, L)
        permuted_pixel_features = pixel_features.permute(0, 2, 1)       # (B, L, C)

        for _ in range(self.n_iters):
            dist_matrix = self.pairwise_dist(pixel_features, spixel_features)    # (B, L', L)
            dist_matrix += mask_dist
            affinity_matrix = (-dist_matrix * self.temperture).softmax(1)
            spixel_features = torch.bmm(affinity_matrix.detach(), permuted_pixel_features)
            spixel_features = spixel_features / affinity_matrix.detach().sum(2, keepdim=True).clamp_(min=1e-16)
            spixel_features = spixel_features.permute(0, 2, 1)
        
        dist_matrix = self.pairwise_dist(pixel_features, spixel_features)
        hard_labels = torch.argmin(dist_matrix, dim=1)

        return spixel_features.permute(0, 2, 1), hard_labels

    def pairwise_dist(self, f1, f2):
        return ((f1 * f1).sum(dim=1).unsqueeze(1)
                + (f2 * f2).sum(dim=1).unsqueeze(2)
                - 2 * torch.einsum("bcm, bcn -> bmn", f2, f1))

    def extra_repr(self):
        return f"num_spixels={self.num_spixels}, n_iters={self.n_iters}"
