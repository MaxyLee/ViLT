from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule, F30KCIPDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule, CocoSubDataModule, CocoSubCIPDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .ve_datamodule import VEDataModule
from .nlvr2_datamodule import NLVR2DataModule
from .cub_datamodule import CubDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "f30k_cip": F30KCIPDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "coco_sub": CocoSubDataModule,
    "coco_sub_cip": CocoSubCIPDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "ve": VEDataModule,
    "nlvr2": NLVR2DataModule,
    "cub": CubDataModule,
}
