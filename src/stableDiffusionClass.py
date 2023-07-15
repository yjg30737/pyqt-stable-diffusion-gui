import gc
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
from transformers import TRANSFORMERS_CACHE

from script import get_info


class StableDiffusionWrapper:
    def __init__(self):
        super(StableDiffusionWrapper, self).__init__()
        self.__initVal()

    def __initVal(self):
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.__device == 'cuda':
            # https://huggingface.co/docs/diffusers/optimization/fp16#use-tf32-instead-of-fp32-on-ampere-and-later-cuda-devices
            # fast speed, less memory, less accuracy
            torch.backends.cuda.matmul.allow_tf32 = True

        self.__model_id = None
        self.__cache_dir = TRANSFORMERS_CACHE
        self.__torch_dtype = torch.float16
        self.__is_safety_checker = True

        self.__pipeline = None

        self.__enable_xformers_memory_efficient_attention = False
        self.__enable_vae_slicing = False
        self.__enable_attention_slicing = False
        self.__enable_vae_tiling = False
        self.__enable_sequential_cpu_offload = False
        self.__enable_model_cpu_offload = False

        self.__lora_path = []

    def init_wrapper(self, model_id, cache_dir=TRANSFORMERS_CACHE, torch_dtype=torch.float16, is_safety_checker=True, sampler='PNDMScheduler'):
        # clear cache to avoid OutOfMemoryError
        gc.collect()
        torch.cuda.empty_cache()

        if self.__model_id != model_id or self.__cache_dir != cache_dir or self.__torch_dtype != torch_dtype or self.__is_safety_checker != is_safety_checker:
            self.__model_id = model_id if self.__model_id != model_id else model_id
            self.__cache_dir = cache_dir if self.__cache_dir != cache_dir else cache_dir
            self.__torch_dtype = torch_dtype if self.__torch_dtype != torch_dtype else torch_dtype
            self.__is_safety_checker = is_safety_checker if self.__is_safety_checker != is_safety_checker else self.__is_safety_checker

            self.__pipeline = StableDiffusionPipeline.from_pretrained(
                self.__model_id, cache_dir=self.__cache_dir, torch_dtype=self.__torch_dtype).to(self.__device)

            self.__set_sampler(sampler)

            if not self.__is_safety_checker:
                self.__pipeline.safety_checker = None

    def __set_sampler(self, sampler):
        for compatible in self.__pipeline.scheduler.compatibles:
            print(compatible)

        if sampler == 'PNDMScheduler':
            self.__pipeline.scheduler = PNDMScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )
        elif sampler == 'DPMSolverMultistepScheduler':
            self.__pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )
        elif sampler == 'DPMSolverSinglestepScheduler':
            self.__pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )
        elif sampler == 'LMSDiscreteScheduler':
            self.__pipeline.scheduler = LMSDiscreteScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )
        elif sampler == 'HeunDiscreteScheduler':
            self.__pipeline.scheduler = HeunDiscreteScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )
        elif sampler == 'EulerDiscreteScheduler':
            self.__pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )
        elif sampler == 'EulerAncestralDiscreteScheduler':
            self.__pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.__pipeline.scheduler.config, use_karras_sigmas=True
            )


    def set_saving_memory_attr(self, enable_xformers_memory_efficient_attention,
                                        enable_vae_slicing,
                                        enable_attention_slicing,
                                        enable_vae_tiling,
                                        enable_sequential_cpu_offload,
                                        enable_model_cpu_offload):
        if self.__enable_xformers_memory_efficient_attention != enable_xformers_memory_efficient_attention:
            self.__enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
            if self.__enable_xformers_memory_efficient_attention:
                self.__pipeline.enable_xformers_memory_efficient_attention()
            else:
                self.__pipeline.disable_xformers_memory_efficient_attention()
        if self.__enable_vae_slicing != enable_vae_slicing:
            self.__enable_vae_slicing = enable_vae_slicing
            if self.__enable_vae_slicing:
                self.__pipeline.enable_vae_slicing()
            else:
                self.__pipeline.disable_vae_slicing()
        if self.__enable_attention_slicing != enable_attention_slicing:
            self.__enable_attention_slicing = enable_attention_slicing
            if self.__enable_attention_slicing:
                self.__pipeline.enable_attention_slicing()
            else:
                self.__pipeline.disable_attention_slicing()
        if self.__enable_vae_tiling != enable_vae_tiling:
            self.__enable_vae_tiling = enable_vae_tiling
            if self.__enable_vae_tiling:
                self.__pipeline.enable_vae_tiling()
            else:
                self.__pipeline.disable_vae_tiling()
        if self.__enable_sequential_cpu_offload != enable_sequential_cpu_offload:
            self.__enable_sequential_cpu_offload = enable_sequential_cpu_offload
            if self.__enable_sequential_cpu_offload:
                self.__pipeline.enable_sequential_cpu_offload()
        if self.__enable_model_cpu_offload != enable_model_cpu_offload:
            self.__enable_model_cpu_offload = enable_model_cpu_offload
            if self.__enable_model_cpu_offload:
                self.__pipeline.enable_model_cpu_offload()

    def load_lora_weights(self, lora_path):
        if lora_path in self.__lora_path:
            pass
        else:
            self.__lora_path.append(lora_path)
            weight_name = get_info(lora_path)[0]
            self.__pipeline.load_lora_weights(lora_path, weight_name=weight_name)

    def get_pipeline(self):
        return self.__pipeline