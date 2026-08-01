from __future__ import annotations

import pytest

from local_lm.worker_failures import (
    WorkerFailureCode,
    classify_worker_failure,
)


def classify(
    *,
    detail: str | None = None,
    tail: str | None = None,
    exit_code: int | None = 1,
) -> WorkerFailureCode | None:
    failure = classify_worker_failure(
        failure_detail=detail,
        stderr_tail=tail,
        exit_code=exit_code,
    )
    return failure.code


@pytest.mark.parametrize(
    "tail",
    [
        # PyTorch, which is what ComfyUI raises through.
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: CUDA out of memory.",
        # ComfyUI's own wording when it cannot place a model.
        "Allocation on device 0 would exceed allowed memory",
        # llama.cpp's CUDA backend.
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4096.00 MiB "
        "on device 0 failed: out of memory",
        "llama_model_load: error loading model: unable to allocate 5000 MiB on the GPU",
        # Apple and AMD paths.
        "MPS backend out of memory (MPS allocated: 9.00 GB)",
        "hipErrorOutOfMemory",
        # The driver giving up entirely.
        "DXGI_ERROR_DEVICE_REMOVED",
    ],
)
def test_a_graphics_memory_exhaustion_is_recognised(tail: str) -> None:
    """The most common way a local model fails to start, and nothing saw it."""
    assert classify(tail=tail) == WorkerFailureCode.OOM_VRAM


@pytest.mark.parametrize(
    ("tail", "exit_code"),
    [
        ("std::bad_alloc", 1),
        ("MemoryError", 1),
        ("OSError: [Errno 12] Cannot allocate memory", 1),
        ("The paging file is too small for this operation to complete", 1),
        ("Not enough memory resources are available to process this command", 1),
        # The Linux OOM killer leaves nothing behind: the process never got to
        # write anything, so only the exit code names it.
        (None, 137),
        ("", 134),
    ],
)
def test_system_memory_exhaustion_is_recognised(tail: str | None, exit_code: int) -> None:
    assert classify(tail=tail, exit_code=exit_code) == WorkerFailureCode.OOM_HOST


@pytest.mark.parametrize(
    "tail",
    [
        "OSError: [Errno 98] Address already in use",
        "[WinError 10048] Only one usage of each socket address is normally permitted",
        "ERROR: port 8188 is already in use",
    ],
)
def test_a_port_conflict_is_recognised(tail: str) -> None:
    assert classify(tail=tail) == WorkerFailureCode.PORT_IN_USE


@pytest.mark.parametrize(
    "tail",
    [
        "llama_model_load: error loading model: unknown model architecture: 'qwen4'",
        "error loading model architecture: unsupported architecture",
        "gguf_init_from_file: invalid magic character",
        "llama_model_load: unsupported GGUF version 5",
        "load_tensors: missing tensor 'blk.0.attn_q.weight'",
        "ERROR: not a valid safetensors file",
    ],
)
def test_an_unreadable_model_is_recognised(tail: str) -> None:
    assert classify(tail=tail) == WorkerFailureCode.MODEL_INCOMPATIBLE


@pytest.mark.parametrize(
    "detail",
    [
        "FileNotFoundError: [WinError 2] The system cannot find the file specified",
        "llama-server is not recognized as an internal or external command",
        "No such file or directory: 'llama-server'",
    ],
)
def test_a_missing_engine_program_is_recognised(detail: str) -> None:
    assert classify(detail=detail) == WorkerFailureCode.EXECUTABLE_MISSING


def test_a_missing_model_file_is_not_read_as_a_missing_engine() -> None:
    """ "No such file or directory" appears for both, so precedence decides.

    llama.cpp prints its own load failure alongside, and the model patterns run
    first - otherwise a missing model would send someone to reinstall a runtime
    that is present and working.
    """
    tail = (
        "llama_model_load_from_file: failed to load model\n"
        "main: error: unable to load model: No such file or directory\n"
    )
    assert classify(tail=tail) == WorkerFailureCode.MODEL_INCOMPATIBLE


def test_a_startup_timeout_is_recognised_from_the_supervisor_message() -> None:
    assert (
        classify(detail="chat worker did not become healthy in time.")
        == WorkerFailureCode.STARTUP_TIMEOUT
    )


def test_an_unexplained_nonzero_exit_is_a_crash_rather_than_a_guess() -> None:
    assert classify(tail="Segmentation fault", exit_code=11) == WorkerFailureCode.CRASHED


def test_nothing_to_go_on_stays_unknown_with_no_remedy() -> None:
    """A confident wrong remedy sends someone to change the wrong setting."""
    failure = classify_worker_failure(failure_detail=None, stderr_tail=None, exit_code=None)
    assert failure.code == WorkerFailureCode.UNKNOWN
    assert failure.remedy is None


def test_a_clean_exit_is_not_reported_as_a_failure() -> None:
    failure = classify_worker_failure(failure_detail="", stderr_tail="", exit_code=0)
    assert failure.code == WorkerFailureCode.UNKNOWN


def test_graphics_memory_wins_over_the_load_failure_it_causes() -> None:
    """Running out of VRAM prints a generic load error immediately afterwards.

    Reporting that as an unreadable model would send someone to reinstall a
    model that is fine, on a machine that simply cannot fit it.
    """
    tail = (
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 6144.00 MiB "
        "on device 0 failed: out of memory\n"
        "llama_model_load: error loading model: failed to allocate buffer\n"
        "llama_init_from_file: failed to load model\n"
    )
    assert classify(tail=tail) == WorkerFailureCode.OOM_VRAM


def test_a_process_killed_while_reporting_a_vram_failure_keeps_that_cause() -> None:
    """Exit 137 is checked after the patterns for exactly this case."""
    assert (
        classify(tail="CUDA out of memory. Tried to allocate 2.00 GiB", exit_code=137)
        == WorkerFailureCode.OOM_VRAM
    )


@pytest.mark.parametrize("code", list(WorkerFailureCode))
def test_every_named_failure_carries_a_remedy_except_unknown(
    code: WorkerFailureCode,
) -> None:
    """A code without advice is a label, which is what we already had."""
    from local_lm.worker_failures import _REMEDIES

    if code == WorkerFailureCode.UNKNOWN:
        assert code not in _REMEDIES
        return
    remedy = _REMEDIES[code]
    assert remedy.endswith(".")
    assert len(remedy) > 40
    # Written for someone who does not know what a tensor is.
    assert not any(word in remedy for word in ("stderr", "traceback", "errno", "PID"))
