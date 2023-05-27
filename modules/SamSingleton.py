import torch
from segment_anything import sam_model_registry, SamPredictor


class SamSingleton:
    _instance = None

    @staticmethod
    def get_instance(path):
        if not SamSingleton._instance:
            DEVICE = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            MODEL_TYPE = "vit_h"

            sam = sam_model_registry[MODEL_TYPE](
                checkpoint=path).to(device=DEVICE)
            SamSingleton._instance = SamPredictor(sam)

        return SamSingleton._instance
