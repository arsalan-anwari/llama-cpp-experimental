# llama-cpp-experimental

## Build your own GGUF from Python dataclasses / pydantic models

The script `convert_dataclass_to_gguf.py` turns structured Python objects (dataclasses or pydantic models) into GGUF files, including tensors.

Supported:
- Metadata values: strings, bools, signed/unsigned ints, floats, arrays of those (flattened from nested structures).
- Tensors: numeric dtypes F16/F32/F64/I8/I16/I32/I64 (or a supported ggml quantization type when pre-packed).

Limitations:
- Arbitrary tensor dtypes (e.g., `uint8`, complex) are rejected.
- Nested objects are flattened to dotted keys; arrays must be homogeneously typed.

Minimal Python example:
```python
from dataclasses import dataclass
import numpy as np
from convert_dataclass_to_gguf import convert_dataclass_to_gguf, ModelArchitecture

@dataclass
class TinyConfig:
    vocab_size: int
    context_length: int

cfg = TinyConfig(vocab_size=32000, context_length=2048)
weights = np.ones((4, 4), dtype=np.float32)

convert_dataclass_to_gguf(
    cfg,
    "models/custom/tiny.gguf",
    architecture=ModelArchitecture.LLAMA,
    tensor_overrides={"tiny.weight": weights},
    description="Tiny demo GGUF built from a dataclass.",
)
```

CLI usage (JSON payload optional):
```bash
python convert_dataclass_to_gguf.py \
  --target my_module:TinyConfig \
  --data payload.json \
  --output models/custom/tiny.gguf \
  --arch llama \
  --description "tiny demo"
```

Inspect GGUF metadata/tensors:
```bash
llama-gguf models/custom/tiny.gguf r n
```

Available commands:
```
usage: convert_dataclass_to_gguf.py [-h] --target TARGET [--data DATA] --output OUTPUT
                                    [--arch {mmproj,llama,llama4,deci,falcon,baichuan,grok,gpt2,gptj,gptneox,mpt,starcoder,refact,bert,nomic_bert,nomic_bert_moe,neo_bert,jina_bert_v2,jina_bert_v3,bloom,stablelm,qwen,qwen2,qwen2moe,qwen2vl,qwen3,qwen3moe,qwen3next,qwen3vl,qwen3vlmoe,phi2,phi3,phimoe,plamo,plamo2,codeshell,orion,internlm2,minicpm,minicpm3,gemma,gemma2,gemma3,gemma3n,gemma_embedding,starcoder2,rwkv6,rwkv6qwen2,rwkv7,arwkv7,mamba,mamba2,jamba,xverse,command_r,cohere2,dbrx,olmo,olmo2,olmoe,openelm,arctic,deepseek,deepseek2,chatglm,glm4,glm4_moe,bitnet,t5,t5encoder,jais,nemotron,nemotron_h,exaone,exaone4,granite,granite_moe,granite_hybrid,chameleon,wavtokenizer_dec,plm,bailingmoe,bailingmoe2,dots1,arcee,afmoe,ernie4_5,ernie4_5_moe,falcon_h1,hunyuan_moe,hunyuan_dense,smollm3,gpt_oss,lfm2,lfm2moe,dream,smallthinker,llada,llada_moe,seed_oss,grovemoe,apertus,minimaxm2,cogvlm,rnd1,pangu_embed,mistral3,dataclass_generic}]
                                    [--metadata-prefix METADATA_PREFIX] [--name NAME] [--description DESCRIPTION] [--tensor NAME=PATH] [--big-endian]
                                    [--use-temp-file]

Serialize dataclass or pydantic models into GGUF.

options:
  -h, --help            show this help message and exit
  --target TARGET       Python path to the dataclass or pydantic model (module:object).
  --data DATA           Path to JSON payload used to instantiate the model.
  --output OUTPUT       Where to write the GGUF file.
  --arch {mmproj,llama,llama4,deci,falcon,baichuan,grok,gpt2,gptj,gptneox,mpt,starcoder,refact,bert,nomic_bert,nomic_bert_moe,neo_bert,jina_bert_v2,jina_bert_v3,bloom,stablelm,qwen,qwen2,qwen2moe,qwen2vl,qwen3,qwen3moe,qwen3next,qwen3vl,qwen3vlmoe,phi2,phi3,phimoe,plamo,plamo2,codeshell,orion,internlm2,minicpm,minicpm3,gemma,gemma2,gemma3,gemma3n,gemma_embedding,starcoder2,rwkv6,rwkv6qwen2,rwkv7,arwkv7,mamba,mamba2,jamba,xverse,command_r,cohere2,dbrx,olmo,olmo2,olmoe,openelm,arctic,deepseek,deepseek2,chatglm,glm4,glm4_moe,bitnet,t5,t5encoder,jais,nemotron,nemotron_h,exaone,exaone4,granite,granite_moe,granite_hybrid,chameleon,wavtokenizer_dec,plm,bailingmoe,bailingmoe2,dots1,arcee,afmoe,ernie4_5,ernie4_5_moe,falcon_h1,hunyuan_moe,hunyuan_dense,smollm3,gpt_oss,lfm2,lfm2moe,dream,smallthinker,llada,llada_moe,seed_oss,grovemoe,apertus,minimaxm2,cogvlm,rnd1,pangu_embed,mistral3,dataclass_generic}
                        Model architecture tag to embed into GGUF metadata.
  --metadata-prefix METADATA_PREFIX
                        Optional prefix for generated metadata keys.
  --name NAME           Overrides general.name metadata.
  --description DESCRIPTION
                        Adds general.description metadata.
  --tensor NAME=PATH    Optional .npy tensor to include. May be provided multiple times.
  --big-endian          Emit GGUF data using big endian encoding.
  --use-temp-file       Buffer tensors to disk before writing.
```

