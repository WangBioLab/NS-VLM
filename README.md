# NS-VLM-I

NS-VLM-I provides standalone Next-State VLM inference.

## Inputs

Every inference call accepts only:

- `prompt`
- `images`: one image or a sequence of images.

## Configuration

All model and runtime settings are defined by `InferenceConfig`. Command-line inference creates this configuration from:

- `NS_VLM_I_CHECKPOINT`
- `NS_VLM_I_MODEL`
- `NS_VLM_I_DEVICE`
- `NS_VLM_I_DTYPE`
- `NS_VLM_I_LOCAL_FILES_ONLY`

## Command-line inference

```text
python infer.py --prompt "What abnormality is visible?" --image /data/image.png
```

For multiple images, repeat `--image`:

```text
python infer.py --prompt "Describe the most likely finding across these views." --image /data/view1.png --image /data/view2.png
```

## Python inference

```python
from ns_vlm_i import InferenceConfig, NSVLMI

config = InferenceConfig(
    checkpoint_path="/models/ns_vlm_i.safetensors",
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    device="auto",
    dtype="bf16",
)
engine = NSVLMI(config)
result = engine(
    "What abnormality is visible?",
    ["/data/view1.png", "/data/view2.png"],
)
print(result.answer)
```
