# CAT (Compress And Attend Transformers) implementation

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.cat.configuration_cat import CATConfig
from fla.models.cat.modeling_cat import CATForCausalLM, CATModel

AutoConfig.register(CATConfig.model_type, CATConfig, exist_ok=True)
AutoModel.register(CATConfig, CATModel, exist_ok=True)
AutoModelForCausalLM.register(CATConfig, CATForCausalLM, exist_ok=True)

__all__ = ['CATConfig', 'CATForCausalLM', 'CATModel']
