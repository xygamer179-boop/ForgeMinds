"""Options for ForgeMind-managed torch.compile calls.

FM's custom ops remain opaque to Inductor; this setting only affects Inductor's
own generated Triton code. Do not mutate the process-wide Inductor configuration.
"""

import torch


def compile_options(backend, mode="default"):
    """Return compile kwargs, disabling block pointers for Inductor when supported.

    PyTorch forbids passing mode and options together. Expand the requested mode
    using PyTorch's own API before applying the per-compile override. Other
    backends retain their original mode and receive no Inductor-specific options.
    """
    if backend != "inductor":
        return {"mode": mode}
    inductor = torch._inductor
    if (not hasattr(inductor, "list_options") or not hasattr(inductor, "list_mode_options")
            or "triton.use_block_ptr" not in inductor.list_options()):
        return {"mode": mode}
    options = dict(inductor.list_mode_options(mode=mode))
    options["triton.use_block_ptr"] = False
    return {"options": options}
