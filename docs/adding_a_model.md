# Adding a New Model Backend

## Step 1 — Create the backend file

Create `indic_eval/models/my_model.py`:

```python
import os
from indic_eval.models.base import BaseModel


class MyModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # initialise your client here
        self.api_key = os.environ["MY_API_KEY"]

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            # call your API / library
            response = your_client.complete(prompt, max_tokens=max_tokens)
            return response.text.strip()
        except Exception as e:
            print(f"MyModel error: {e}")
            return ""
```

## Step 2 — Register a shortname

In `indic_eval/models/factory.py`, add to `MODEL_MAP`:

```python
from indic_eval.models.my_model import MyModel

MODEL_MAP["my-model-name"] = ("custom", "provider/model-id")
```

And add a branch in `get_model()`:

```python
elif backend == "custom":
    return MyModel(model_id)
```

## Step 3 — Add to .env.example

```bash
MY_API_KEY=xxxxxxxxxxxx
```

## Step 4 — Add a config

`configs/my_model_hindi.yaml`:

```yaml
model: my-model-name
lang: hi
tasks: qa,nli
num_samples: 100
output_dir: results/
```

## Step 5 — Test it

```bash
python -m indic_eval.cli evaluate --model my-model-name --lang hi --tasks qa --num-samples 5
```
