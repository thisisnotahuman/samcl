from samcl.models.unimodal_dual_encoder import (
    UniModalDualEncoder,
    UniModalDualEncoderConfig,
    load_unimodal_processors,
)

__all__ = [
    "UniModalDualEncoder",
    "UniModalDualEncoderConfig",
    "load_unimodal_processors",
]

# Optional: CVCL student (requires extra deps like huggingface_hub).
try:
    from samcl.models.cvcl_dual_encoder import CvclDualEncoder, CvclDualEncoderConfig  # noqa: F401

    __all__ += ["CvclDualEncoder", "CvclDualEncoderConfig"]
except Exception:
    pass

