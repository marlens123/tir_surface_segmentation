from segmentation_models_pytorch.base.model import SegmentationModel as SegmentationModel_SMP
import torch

@torch.jit.unused
def is_torch_compiling():
    try:
        return torch.compiler.is_compiling()
    except Exception:
        try:
            import torch._dynamo as dynamo  # noqa: F401

            return dynamo.is_compiling()
        except Exception:
            return False

class SegmentationModel(SegmentationModel_SMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x, return_features: bool = False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if return_features:
            return masks, features
        return masks