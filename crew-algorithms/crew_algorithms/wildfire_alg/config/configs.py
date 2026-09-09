from attrs import define    

@define(auto_attribs=True)

class Kimi1LLMConfig:
    actor_model: str = "moonshotai/Kimi-K2.6"
    critic_model: str = "moonshotai/Kimi-K2.6"
    planner_model: str = "moonshotai/Kimi-K2.6"
    translator_model: str = "moonshotai/Kimi-K2.6"

    large_model: str = "moonshotai/Kimi-K2.6"
    small_model: str = "moonshotai/Kimi-K2.6"
    reasoning_model: str = "moonshotai/Kimi-K2.6"

    api_key: str = "KIMI_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

class KimiLLMConfig:
    actor_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    critic_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    planner_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    translator_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

    large_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    small_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    reasoning_model: str = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

    api_key: str = "KIMI_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

class MinimaxLLMConfig:
    actor_model: str = "nvidia/MiniMax-M2.7-NVFP4"
    critic_model: str = "nvidia/MiniMax-M2.7-NVFP4"
    planner_model: str = "nvidia/MiniMax-M2.7-NVFP4"
    translator_model: str = "nvidia/MiniMax-M2.7-NVFP4"

    large_model: str = "nvidia/MiniMax-M2.7-NVFP4"
    small_model: str = "nvidia/MiniMax-M2.7-NVFP4"
    reasoning_model: str = "nvidia/MiniMax-M2.7-NVFP4"

    api_key: str = "MINIMAX_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

class NemotronLLMConfig:
    actor_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    critic_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    planner_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    translator_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    large_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    small_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    reasoning_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    api_key: str = "NEMOTRON_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

class LlamaLLMConfig:
    actor_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    critic_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    planner_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    translator_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    large_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    small_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    reasoning_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    api_key: str = "LLAMA_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

@define(auto_attribs=True)
class GlmLLMConfig:
    actor_model: str = "zai-org/GLM-5.1-FP8"
    critic_model: str = "zai-org/GLM-5.1-FP8"
    planner_model: str = "zai-org/GLM-5.1-FP8"
    translator_model: str = "zai-org/GLM-5.1-FP8"

    large_model: str = "zai-org/GLM-5.1-FP8"
    small_model: str = "zai-org/GLM-5.1-FP8"
    reasoning_model: str = "zai-org/GLM-5.1-FP8"

    api_key: str = "GLM_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

@define(auto_attribs=True)
class GemmaLLMConfig:
    actor_model: str = "google/gemma-4-31B-it"
    critic_model: str = "google/gemma-4-31B-it"
    planner_model: str = "google/gemma-4-31B-it"
    translator_model: str = "google/gemma-4-31B-it"

    large_model: str = "google/gemma-4-31B-it"
    small_model: str = "google/gemma-4-31B-it"
    reasoning_model: str = "google/gemma-4-31B-it"

    api_key: str = "GEMMA_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True


@define(auto_attribs=True)
class DeepseekLLMConfig:
    actor_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    critic_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    planner_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    translator_model: str = "deepseek-ai/DeepSeek-V4-Pro"

    large_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    small_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    reasoning_model: str = "deepseek-ai/DeepSeek-V4-Pro"

    api_key: str = "DEEPSEEK_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True
    
class ErnieLLMConfig:
    actor_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"
    critic_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"
    planner_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"
    translator_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"

    large_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"
    small_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"
    reasoning_model: str = "baidu/ERNIE-4.5-21B-A3B-PT"

    api_key: str = "ERNIE_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

class QwenLLMConfig:
    actor_model: str = "Qwen/Qwen3.6-35B-A3B"
    critic_model: str = "Qwen/Qwen3.6-35B-A3B"
    planner_model: str = "Qwen/Qwen3.6-35B-A3B"
    translator_model: str = "Qwen/Qwen3.6-35B-A3B"

    large_model: str = "Qwen/Qwen3.6-35B-A3B"
    small_model: str = "Qwen/Qwen3.6-35B-A3B"
    reasoning_model: str = "Qwen/Qwen3.6-35B-A3B"

    api_key: str = "QWEN_API_KEY"

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True

@define(auto_attribs=True)
class LLMConfig:
    actor_model: str = "gpt-5.5"
    critic_model: str = "gpt-5.5"
    planner_model: str = "gpt-5.5"
    translator_model: str = "gpt-5.5"

    large_model: str = "gpt-5.5"
    small_model: str = "gpt-5.5"
    reasoning_model: str = "o1-mini"

    api_key: str = "OPENAI_API_KEY"


    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200

    # VLM comparison data collection
    collect_vlm_data: bool = False
    vlm_data_folder: str = "vlm_comparison_data"

    # Team structure generation LLM parameters
    structure_generator_temperature: float = 0.0
    use_structure_critic: bool = True



