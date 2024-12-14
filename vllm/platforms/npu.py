import psutil
import torch

from .interface import Platform, PlatformEnum


class NpuPlatform(Platform):
    _enum = PlatformEnum.NPU

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "npu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()
