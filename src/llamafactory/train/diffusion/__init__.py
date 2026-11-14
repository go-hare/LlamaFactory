__all__ = ["run_diffusion"]


def __getattr__(name):
    if name == "run_diffusion":
        from .workflow import run_diffusion

        return run_diffusion
    raise AttributeError(name)
