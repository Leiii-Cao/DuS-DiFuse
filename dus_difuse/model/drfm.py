"""Detail-Restoration Fidelity Module (DRFM) from Sec. III-D."""

from torch import nn

from .vae import AttentionResidualBlock, SFTBlock


class DetailRestorationFidelityBlock(nn.Module):
    """Inject attention-enhanced source features through spatial modulation."""

    def __init__(self, decoder_channels: int, fidelity_channels: int) -> None:
        super().__init__()
        self.decoder_channels = decoder_channels
        self.fidelity_channels = fidelity_channels
        self.attention_block = AttentionResidualBlock(fidelity_channels)
        self.sft_block = SFTBlock(decoder_channels, fidelity_channels)

    def forward(self, source_features, decoder_features):
        source_features = self.attention_block(source_features)
        return self.sft_block(source_features, decoder_features)


class DetailRestorationFidelityModule(nn.ModuleList):
    """Multi-scale DRFM blocks aligned with the frozen VAE decoder."""

    def __init__(self, ddconfig) -> None:
        super().__init__()
        channels = ddconfig["ch"]
        for multiplier in ddconfig["ch_mult"]:
            decoder_channels = channels * multiplier
            self.append(
                DetailRestorationFidelityBlock(
                    decoder_channels=decoder_channels,
                    fidelity_channels=decoder_channels * 2,
                )
            )
