import math
from typing import Union

import numpy as np
import torch
from einops import rearrange
from torch import nn


class Patchify2D(nn.Module):
    """Turn an image into patches of patch_size x patch_size.

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches

    Returns
    -------
    torch.Tensor
        Tensor of patches with shape [N, P, D] where:
        - N is number of patches (H/patch_size_height * W/patch_size_width)
        - P is number of pixels per patch (patch_size_height * patch_size_width)
        - D is the input channel dimension: 3 for RGB images
    """

    def __init__(
        self,
        patch_size: Union[int, tuple[int, int]] = 16,
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    # TODO Change this to hwc if unfold allows for it
    def __call__(self, image_CHW: torch.Tensor) -> torch.Tensor:
        C, H, W = image_CHW.shape

        grid_height = H // self.patch_size[0]
        grid_width = W // self.patch_size[1]
        assert (grid_height * self.patch_size[0]) == H and (
            grid_width * self.patch_size[1]
        ) == W

        image_patched_flat = self.unfold(image_CHW).T
        # NOTE Unflatten patches to recover original shape
        image_patched = rearrange(
            image_patched_flat, "(gh gw) d -> gh gw d", gh=grid_height, gw=grid_width
        )
        return image_patched


class Patchify3D(nn.Module):
    """Patchify a video tensor into tubelets with a certain patch size, similar to 3D convolutions.

    This module converts a video tensor into spatiotemporal patches (tubelets) by:
    1. Grouping frames into temporal chunks of size tubelet_size
    2. Within each chunk, extracting spatial patches of size patch_size x patch_size

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of spatial patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches
    tubelet_size : int, default=2
        Number of consecutive frames to group into each tubelet

    Returns
    -------
    torch.Tensor
        Tensor of tubelets with shape [T, H, W, C] where:
        - T is number of frames (original T/tubelet_size)
        - H,W are spatial grid dimensions (original H,W/patch_size)
        - C is channel dimension (original C * patch_size^2 * tubelet_size)
    """

    def __init__(
        self, patch_size: Union[int, tuple[int, int]] = 16, tubelet_size: int = 2
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.tubelet_size = tubelet_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    # TODO Change this to thwc
    def __call__(self, video_TCHW: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video_TCHW.shape

        timesteps: int = T // self.tubelet_size
        assert (timesteps * self.tubelet_size) == T

        grid_height = H // self.patch_size[0]
        grid_width = W // self.patch_size[1]
        assert (grid_height * self.patch_size[0]) == H and (
            grid_width * self.patch_size[1]
        ) == W

        video_tubed = rearrange(
            video_TCHW, "(n t) c h w -> n (t c) h w", n=timesteps, t=self.tubelet_size
        )

        video_patched_flattened = self.unfold(video_tubed)
        video_patched = rearrange(
            video_patched_flattened,
            "n (t c ph pw) (gh gw) -> (n t) gh gw (c ph pw)",
            t=self.tubelet_size,
            c=C,
            gh=grid_height,
            gw=grid_width,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )

        return video_patched


class TubeMask:
    """Apply tube masking to spatiotemporal video data by masking aligned spatial patches across time.

    This class implements tube masking as used in V-JEPA and similar architectures. It can handle:
    1. Raw video tensors [T, H, W, C]
    2. Pre-patchified tensors where H,W represent a grid of patches

    For example, given:
    - Raw video: [16, 224, 224, 3]
    - Patchified video: [16, 14, 14, 768] (using 16x16 patches)
    The masking pattern is consistent across the temporal dimension, creating "tubes".

    Parameters
    ----------
    ratio : float
        Ratio of patches to mask out (between 0 and 1)
    patch_size : Union[tuple[int, int], int]
        Size of patches for masking. For pre-patchified input, use (1,1)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors containing:
        1. Kept patches with shape [T, N_kept, C]
        2. Masked patches with shape [T, N_masked, C]
        where N_kept + N_masked = H*W/patch_size^2
    """

    def __init__(
        self,
        ratio: float,
        patch_size: Union[tuple[int, int], int],
    ):
        super(TubeMask, self).__init__()
        self.ratio = ratio
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.patch_size = patch_size

    def sample_spatial_mask(
        self, ratio: float, num_spatial_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate spatial masking pattern to be applied across temporal dimension.

        Parameters
        ----------
        ratio : float
            Ratio of patches to mask (between 0 and 1)
        num_spatial_patches : int
            Total number of spatial patches (H*W/patch_size^2)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Indices of patches to keep and discard
        """
        num_keep_spatial = int(num_spatial_patches * (1.0 - ratio))
        mask = np.hstack(
            [
                np.zeros(num_spatial_patches - num_keep_spatial),
                np.ones(num_keep_spatial),
            ]
        )
        np.random.shuffle(mask)
        mask = torch.tensor(mask)
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()
        return mask_keep, mask_discard

    def __call__(self, video_thwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply tube masking to input video.

        Parameters
        ----------
        video_thwc : torch.Tensor
            Input video tensor in [T, H, W, C] format
            Can be either raw video or pre-patchified

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches and masked patches
            Both have shape [T, N, C] where N varies based on ratio
        """
        T, H, W, C = video_thwc.shape
        num_patches_spatial: int = (H // self.patch_size[0]) * (W // self.patch_size[1])
        mask_keep, mask_discard = self.sample_spatial_mask(
            self.ratio, num_patches_spatial
        )
        mask_keep, mask_discard = (
            mask_keep.unsqueeze(-1).expand(T, -1, C),
            mask_discard.unsqueeze(-1).expand(T, -1, C),
        )
        # Flatten video across the H,W dimensions
        video_flattened_grid = rearrange(video_thwc, "t h w c -> t (h w) c")
        # since the masks contain indices to keep, we can use gather to apply the masking:
        masked_video_keep = torch.gather(video_flattened_grid, dim=1, index=mask_keep)
        masked_video_discard = torch.gather(
            video_flattened_grid, dim=1, index=mask_discard
        )
        return masked_video_keep, masked_video_discard


class MultiBlock3DMask:
    """Apply multi-block 3D masking to spatiotemporal video data.

    This implements the masking strategy from JEPA, which generates multiple block masks
    with configurable spatial and temporal scales, aspect ratios, and number of blocks.

    Parameters
    ----------
    spatial_scale : tuple[float, float]
        Min and max scale for spatial masking (e.g. (0.2, 0.8))
    temporal_scale : tuple[float, float]
        Min and max scale for temporal masking (e.g. (1.0, 1.0))
    aspect_ratio : tuple[float, float]
        Min and max aspect ratios for blocks (e.g. (0.3, 3.0))
    num_blocks : int
        Number of mask blocks to generate
    max_temporal_keep : float
        Maximum ratio of temporal frames to keep (1.0 = all frames)
    patch_size : Union[tuple[int, int], int]
        Size of patches to mask
    """

    def __init__(
        self,
        spatial_scale: tuple[float, float] = (0.2, 0.8),
        temporal_scale: tuple[float, float] = (1.0, 1.0),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        num_blocks: int = 1,
        max_temporal_keep: float = 1.0,
        patch_size: Union[tuple[int, int], int] = (16, 16),
    ):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2
        else:
            self.patch_size = patch_size

        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.max_temporal_keep = max_temporal_keep

    def _sample_block_size(
        self,
        height: int,
        width: int,
        duration: int,
        generator: torch.Generator,
    ) -> tuple[int, int, int]:
        """Sample a random block size given constraints."""
        min_t, max_t = self.temporal_scale
        t_scale = min_t + torch.rand(1, generator=generator).item() * (max_t - min_t)
        t = max(1, int(duration * t_scale))

        # Sample spatial block size
        min_s, max_s = self.spatial_scale
        s_scale = min_s + torch.rand(1, generator=generator).item() * (max_s - min_s)
        spatial_num_keep = int(height * width * s_scale)

        # Sample aspect ratio
        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + torch.rand(1, generator=generator).item() * (max_ar - min_ar)

        # Calculate block height/width
        h = int(round(math.sqrt(spatial_num_keep * ar)))
        w = int(round(math.sqrt(spatial_num_keep / ar)))
        h = min(h, height)
        w = min(w, width)

        return (t, h, w)

    def _sample_block_mask(
        self,
        block_size: tuple[int, int, int],
        height: int,
        width: int,
        duration: int,
    ) -> torch.Tensor:
        """Generate a single block mask."""
        t, h, w = block_size
        top = torch.randint(0, height - h + 1, (1,))
        left = torch.randint(0, width - w + 1, (1,))
        start = torch.randint(0, duration - t + 1, (1,))

        mask = torch.ones((duration, height, width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        max_context_duration = max(1, int(duration * self.max_temporal_keep))
        if max_context_duration < duration:
            mask[max_context_duration:, :, :] = 0

        return mask

    def __call__(self, video_thwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to input video. An image tensor can be passed as well, in which case the mask will interpret it as a single frame.

        Parameters
        ----------
        video_thwc : torch.Tensor
            Input video tensor [T, H, W, C]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches [T, N_kept, C] and masked patches [T, N_masked, C]
        """
        # NOTE Add temporal dimension for image
        is_image = video_thwc.ndim == 3
        if video_thwc.ndim == 3:
            video_thwc = video_thwc.unsqueeze(0)

        T, H, W, C = video_thwc.shape
        grid_h = H // self.patch_size[0]
        grid_w = W // self.patch_size[1]

        block_size = self._sample_block_size(grid_h, grid_w, T, torch.Generator())

        mask = torch.ones((T, grid_h, grid_w), dtype=torch.int32)
        for _ in range(self.num_blocks):
            mask *= self._sample_block_mask(block_size, grid_h, grid_w, T)

        mask = mask.flatten()
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()

        video_flat = rearrange(video_thwc, "t h w c -> (t h w) c")

        kept = video_flat[mask_keep].reshape(T, -1, C)
        masked = video_flat[mask_discard].reshape(T, -1, C)

        # remove temporal dimension for image
        if is_image:
            kept = kept.squeeze(0)
            masked = masked.squeeze(0)

        return kept, masked


if __name__ == "__main__":
    tube = TubeMask(0.5, (1, 1))
    mb3d = MultiBlock3DMask((0.2, 0.8), (1.0, 1.0), (0.3, 3.0), 1, 1.0, (1, 1))
    patchify = Patchify3D(16, 2)
    patchify2d = Patchify2D(16)
    randvid = torch.randn((16, 3, 224, 224))
    randimg = torch.randn((3, 224, 224))
    # 16, 14, 14, 768
    vidpatch = patchify(randvid)
    imgpatch = patchify2d(randimg)
    # since ratio is 0.5: 16, 98, 768
    vidtubemask = tube(vidpatch)
    vidmb3dmask = mb3d(vidpatch)
    imgmb3dmask = mb3d(imgpatch)
