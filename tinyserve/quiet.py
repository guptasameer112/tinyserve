"""Keep notebook output readable. Silences library chatter only; real errors still show."""
import warnings


def quiet_logs():
    """HF hub auth/progress noise and transformers warnings (e.g. max_length vs max_new_tokens)."""
    # (?s) so '.' also matches the leading newline in the message
    warnings.filterwarnings("ignore", message=r"(?s).*HF_TOKEN")
    try:
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as hub_logging

        disable_progress_bars()
        hub_logging.set_verbosity_error()
    except ImportError:
        pass
