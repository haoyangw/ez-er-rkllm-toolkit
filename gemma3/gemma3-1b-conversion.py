# Requires rkllm-toolkit Python wheel to be installed
from rkllm.api import RKLLM
from transformers import Gemma3ForCausalLM, AutoTokenizer
import safetensors
import torch

# Unsloth version of Gemma 3 1B Instruct
modelpath = 'unsloth/gemma-3-1b-it'

model = Gemma3ForCausalLM.from_pretrained(modelpath, device_map='cpu', torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(modelpath)

model.save_pretrained('llm')
tokenizer.save_pretrained('llm')

del model
model = None
del tokenizer
tokenizer = None

modelpath = 'llm'
savepath = 'llm/gemma-3-1b-it-w8a8.rkllm'

llm = RKLLM()

ret = llm.load_huggingface(model=modelpath, device='cpu')
if ret != 0:
    print('Load model failed!')
    exit(ret)


ret = llm.build(
        do_quantization=True, 
        optimization_level=0, 
        quantized_dtype='w8a8', 
        # hybrid ratio of 25% gives a good balance
        hybrid_rate=0.25,
        # 16k, the max context length for rkllm-toolkit(Gemma 3 1B supports 32k max)
        max_context=4096 * 4,
        quantized_algorithm='normal', 
        target_platform='rk3588', 
        num_npu_core=3, 
        extra_qparams=None, 
        dataset=None
        )
if ret != 0:
    print('Build model failed!')
    exit(ret)

ret = llm.export_rkllm(savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)
