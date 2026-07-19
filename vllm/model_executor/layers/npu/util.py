import os
import torch
import torch_npu
import acl
from .py_npu_ops import NPUPtr, DataType


NPU_DT_MAPPING = {
    torch.uint8: DataType.DT_UINT8,
    torch.int8: DataType.DT_INT8,
    torch.int32: DataType.DT_INT32,
    torch.float16: DataType.DT_FLOAT16,
    torch.bfloat16: DataType.DT_BFLOAT16,
    torch.float32: DataType.DT_FLOAT32,
    torch.int64: DataType.DT_INT64
}

def to_npu_dtype(torch_dt):
    return NPU_DT_MAPPING[torch_dt]


_cached_default_stream = None


def get_default_stream():
    """Default NPU stream handle (cached).

    torch.npu.default_stream() re-resolves the device index through
    torch._utils._get_device_index (which even consults
    torch.cuda.is_available) on EVERY call — ~120us each, and it is called
    ~3000x per decode token. The default stream of the device does not change,
    so cache it after the first lookup.
    """
    global _cached_default_stream
    s = _cached_default_stream
    if s is None:
        s = torch.npu.default_stream().npu_stream
        _cached_default_stream = s
    return s


def get_pointer(x):
    return NPUPtr(x.data_ptr())


class MSTX:

    def __init__(self, msg):
        self.stamp = acl.prof.create_stamp()
        self.msg = msg

    def __enter__(self):
        acl.prof.set_stamp_trace_message(self.stamp, self.msg, len(self.msg))
        acl.prof.push(self.stamp)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        acl.prof.pop(self.stamp)
        acl.prof.destroy_stamp(self.stamp)



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

def dump_tensor(output_path, tensor):
    vt = tensor
    if vt.dtype == torch.bfloat16:
        vt = vt.view(torch.half)
    np_tensor = vt.cpu().numpy()
    dump_path = os.path.join("/data/debug", output_path)
    np_tensor.tofile(dump_path)
    print(f"dump_tensor to {dump_path}, tensor shape: {tensor.shape} dtype: {tensor.dtype}")
    

