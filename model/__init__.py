# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from .utils import cast_to

from .dpt_head import DPTHead
from .sdt_head import SDTHead
from .encoder import BackboneLayersSet, DinoVisionTransformerWrapper, PatchSizeAdaptationStrategy


class FeaturesToDepth(torch.nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=100,
        bins_strategy="linear",
        norm_strategy="linear",
    ):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        output = torch.relu(x) + self.min_depth
        return output


def make_head(
    embed_dims: int | list[int],
    n_output_channels: int,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    head_type: str = "linear",
    **kwargs,
) -> torch.nn.Module:

    if isinstance(embed_dims, int):
        embed_dims = [embed_dims]

    if head_type == "linear":
        raise NotImplementedError
    elif head_type == "dpt":
        decoder = DPTHead(
            in_channels=embed_dims,
            n_output_channels=n_output_channels,
            readout_type="project" if use_cls_token else "ignore",
            use_batchnorm=use_batchnorm,
            **kwargs,
        )
    elif head_type == "sdt":
        decoder = SDTHead(
            in_channels=embed_dims,
            n_output_channels=n_output_channels,
            use_cls_token=use_cls_token,
            **kwargs
        )
    else:
        raise NotImplementedError("only linear, DPT and SDT head supported")
    return decoder


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_dtype=torch.float,
        decoder_dtype=torch.float,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_dtype = encoder_dtype
        self.decoder_dtype = decoder_dtype
        self.is_cuda = torch.cuda.is_available()

    def forward(self, x):
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=False):
            x = x.to(self.encoder_dtype)
            x = self.encoder(x)
            x = cast_to(x, self.decoder_dtype)
        return self.decoder(x)


def build_depther(
    backbone: torch.nn.Module,
    backbone_out_layers: list[int] | BackboneLayersSet,
    n_output_channels: int,
    use_backbone_norm: bool = False,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    adapt_to_patch_size: PatchSizeAdaptationStrategy = PatchSizeAdaptationStrategy.CENTER_PADDING,
    head_type: str = "dpt",
    encoder_dtype: torch.dtype = torch.float,
    decoder_dtype: torch.dtype = torch.float,
    min_depth: float = 0.001,
    max_depth: float = 100,
    bins_strategy: str = "linear",
    norm_strategy: str = "linear",
    **kwargs,
):
    encoder = DinoVisionTransformerWrapper(
        backbone_model=backbone,
        backbone_out_layers=backbone_out_layers,
        use_backbone_norm=use_backbone_norm,
        adapt_to_patch_size=adapt_to_patch_size,
    )
    encoder = encoder.to(encoder_dtype)
    # encoder.requires_grad_(True)
    # encoder.train()
    encoder.requires_grad_(False)
    encoder.eval()

    decoder = make_head(
        encoder.embed_dims,
        n_output_channels=n_output_channels,
        use_batchnorm=use_batchnorm,
        use_cls_token=use_cls_token,
        head_type=head_type,
        **kwargs,
    )
    decoder.train()

    features_to_depth = FeaturesToDepth(
        min_depth=min_depth,
        max_depth=max_depth,
        bins_strategy=bins_strategy,
        norm_strategy=norm_strategy,
    )

    return torch.nn.Sequential(
        EncoderDecoder(
            encoder,
            decoder,
            encoder_dtype=encoder_dtype,
            decoder_dtype=decoder_dtype,
        ),
        features_to_depth,
    )
