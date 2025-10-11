# utils/models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_models(gen_model_name: str, critic_model_name: str, device=None):
    """
    Loads generator and critic models with their tokenizers.

    Parameters
    ----------
    gen_model_name : str
        Hugging Face model ID for the text generator.
    critic_model_name : str
        Hugging Face model ID for the critic (alignment evaluator).
    device : str or torch.device
        'cuda', 'mps', or 'cpu'. If None, chosen automatically.

    Returns
    -------
    dict containing:
        {
            "gen_model": model,
            "gen_tok": tokenizer,
            "crit_model": model,
            "crit_tok": tokenizer,
            "device": torch.device
        }
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    print(f"🧠 Using device: {device}")

    # --- Load generator ---
    gen_tok = AutoTokenizer.from_pretrained(gen_model_name)
    gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
    gen_model.config.pad_token_id = gen_tok.eos_token_id

    # --- Load critic ---
    crit_tok = AutoTokenizer.from_pretrained(critic_model_name)
    crit_tok.pad_token = crit_tok.eos_token
    crit_model = AutoModelForCausalLM.from_pretrained(
        critic_model_name,
        torch_dtype=torch.float32,      # safe for phi, neo, etc.
        low_cpu_mem_usage=True
    ).to(device)
    crit_model.config.pad_token_id = crit_tok.eos_token_id

    return {
        "gen_model": gen_model,
        "gen_tok": gen_tok,
        "crit_model": crit_model,
        "crit_tok": crit_tok,
        "device": device
    }
