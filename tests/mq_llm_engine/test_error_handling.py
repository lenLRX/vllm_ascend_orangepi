import asyncio
import tempfile
import time
import uuid
from unittest.mock import Mock

import pytest

from tests.mq_llm_engine.utils import RemoteMQLLMEngine
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing import MQEngineDeadError
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.lora.request import LoRARequest
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

MODEL = "Qwen/Qwen2-0.5B-Instruct"
ENGINE_ARGS = AsyncEngineArgs(model=MODEL)
RAISED_ERROR = KeyError("foo")


@pytest.fixture(scope="function")
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f"ipc://{td}/{uuid.uuid4()}"


def run_with_evil_forward(engine_args: AsyncEngineArgs, ipc_path: str):
    # Make engine.
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.UNKNOWN_CONTEXT,
        ipc_path=ipc_path)

    # Raise error during first forward pass.
    engine.engine.model_executor.execute_model = Mock(side_effect=RAISED_ERROR)

    # Run engine.
    engine.start()


@pytest.mark.asyncio
async def test_evil_forward(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket,
                           run_fn=run_with_evil_forward) as engine:

        client = await engine.make_client()

        # Fast health probe.
        fast_health_probe_task = asyncio.create_task(
            client.run_check_health_loop(timeout=1.0))

        # Server should be healthy after initial probe.
        await asyncio.sleep(2.0)
        await client.check_health()

        # Throws an error in first forward pass.
        try:
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(),
                                           request_id=uuid.uuid4()):
                pass
        except Exception as e:
            # First exception should be a RAISED_ERROR
            assert repr(e) == repr(RAISED_ERROR)
            assert client.errored

        # Engine is errored, should get ENGINE_DEAD_ERROR.
        try:
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(),
                                           request_id=uuid.uuid4()):
                pass
        except Exception as e:
            # Next exception should be an ENGINE_DEAD_ERROR
            assert client.errored, "Client should be dead."
            assert isinstance(e, MQEngineDeadError), (
                "Engine should be dead and raise ENGINE_DEAD_ERROR")

        await asyncio.sleep(2.0)
        try:
            await client.check_health()
        except Exception as e:
            assert repr(e) == repr(RAISED_ERROR), (
                "Health check raise the original error.")

        # Cleanup
        await fast_health_probe_task
        client.close()


def run_with_evil_model_executor_health(engine_args: AsyncEngineArgs,
                                        ipc_path: str):
    # Make engine.
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.UNKNOWN_CONTEXT,
        ipc_path=ipc_path)

    # Raise error during first forward pass.
    engine.engine.model_executor.check_health = Mock(side_effect=RAISED_ERROR)

    # Run engine.
    engine.start()


@pytest.mark.asyncio
async def test_failed_health_check(tmp_socket):
    with RemoteMQLLMEngine(
            engine_args=ENGINE_ARGS,
            ipc_path=tmp_socket,
            run_fn=run_with_evil_model_executor_health) as engine:

        client = await engine.make_client()
        assert client.is_running

        # Health probe should throw RAISED_ERROR.
        await asyncio.sleep(10)
        try:
            await client.check_health()
        except Exception as e:
            assert client.errored, "Client should be dead."
            assert repr(e) == repr(RAISED_ERROR), (
                "Health check raise the original error.")

        # Generate call should throw ENGINE_DEAD_ERROR
        try:
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(),
                                           request_id=uuid.uuid4()):
                pass
        except Exception as e:
            assert client.errored, "Client should be dead."
            assert isinstance(e, MQEngineDeadError), (
                "Engine should be dead and raise ENGINE_DEAD_ERROR")

        # Cleanup
        client.close()


def run_with_evil_abort(engine_args: AsyncEngineArgs, ipc_path: str):
    # Make engine.
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.UNKNOWN_CONTEXT,
        ipc_path=ipc_path)

    # Raise error during abort call.
    engine.engine.abort_request = Mock(side_effect=RAISED_ERROR)

    # Run engine.
    engine.start()


@pytest.mark.asyncio
async def test_failed_abort(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket,
                           run_fn=run_with_evil_abort) as engine:

        client = await engine.make_client()
        assert client.is_running

        # Firsh check health should work.
        await client.check_health()

        # Trigger an abort on the client side.
        async def bad_abort_after_2s():
            await asyncio.sleep(2.0)
            await client.abort(request_id="foo")

            # Immediately should trigger error.
            try:
                await client.check_health()
            except Exception as e:
                assert client.errored, "Client should be dead."
                assert repr(e) == repr(RAISED_ERROR), (
                    "Health check raise the original error.")

        # Trigger an abort in 2s from now.
        abort_task = asyncio.create_task(bad_abort_after_2s())

        # Exception in abort() will happen during this generation.
        # This will kill the engine and should return ENGINE_DEAD_ERROR.
        try:
            async for _ in client.generate(
                    inputs="Hello my name is",
                    sampling_params=SamplingParams(max_tokens=2000),
                    request_id=uuid.uuid4()):
                pass
        except Exception as e:
            print(f"error is: {e}")
            # Next exception should be an ENGINE_DEAD_ERROR
            assert isinstance(e, MQEngineDeadError), (
                "Engine should be dead and raise ENGINE_DEAD_ERROR")
            assert client.errored

        await abort_task

        client.close()


@pytest.mark.asyncio
async def test_bad_request(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket) as engine:

        client = await engine.make_client()

        # This should fail, but not crash the server.
        try:
            print("calling first generate")
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(),
                                           request_id="abcd-1",
                                           lora_request=LoRARequest(
                                               "invalid-lora", 1,
                                               "invalid-path")):
                pass
        except Exception as e:
            print("got exception")
            assert isinstance(e, ValueError), (
                "Expected ValueError when a LoRARequest in llm_engine")

        # This request should be okay.
        async for _ in client.generate(inputs="Hello my name is",
                                       sampling_params=SamplingParams(),
                                       request_id="abcd-2"):
            pass

        # Confirm server is still running.
        await asyncio.sleep(10.)
        await client.check_health()

        # Shutdown.
        client.close()


@pytest.mark.asyncio
async def test_mp_crash_detection():

    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])
    # use an invalid tensor_parallel_size to trigger the
    # error in the server
    args.tensor_parallel_size = 65536

    start = time.perf_counter()
    async with build_async_engine_client(args):
        pass
    end = time.perf_counter()

    assert end - start < 60, ("Expected vLLM to gracefully shutdown in <60s "
                              "if there is an error in the startup.")


@pytest.mark.asyncio
async def test_mp_cuda_init():
    # it should not crash, when cuda is initialized
    # in the API server process
    import torch
    torch.cuda.init()
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])

    async with build_async_engine_client(args):
        pass