import torch
import torch_npu
import acl
from .py_npu_ops import NPUPtr, DataType


def get_default_stream():
    return torch.npu.default_stream().npu_stream


def get_pointer(x):
    return NPUPtr(x.data_ptr())

class NPUTimer:

    def __init__(self, stream):
        self.stream = stream
        self.duration = 0

    def __enter__(self):
        self.start_event, _ = acl.rt.create_event()
        acl.rt.record_event(self.start_event, self.stream)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        end_event, _ = acl.rt.create_event()
        acl.rt.record_event(end_event, self.stream)
        acl.rt.synchronize_event(end_event)
        self.duration, _ = acl.rt.event_elapsed_time(self.start_event, end_event)
        acl.rt.destroy_event(self.start_event)
        acl.rt.destroy_event(end_event)



