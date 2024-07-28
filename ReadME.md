# SPT: A Lightweight Language Model

NanoLlama is a compact language model trained on Sherlock Holmes stories.

## Model Details

- **Model Type**: NanoLlama (Causal Language Model)
- **Number of Layers**: 12
- **Hidden Size**: 512
- **Number of Attention Heads**: 16
- **Number of KV Heads**: 16
- **Intermediate Size**: 2048
- **Maximum Sequence Length**: 2048
- **Vocabulary Size**: 97 (including special tokens)

## Usage

You can use this model with the Hugging Face Transformers library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("imdatta0/spt")
model = AutoModelForCausalLM.from_pretrained("imdatta0/spt")

# Generate text
input_text = "Sherlock and I were "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## Training

This model was trained on Sherlock Holmes' stories on a single A100 with a batch size of 2 and gradient accumulation steps of 32 effective batch size of 64. It was trained on 1024 length character sequences for 10000 steps.  

## Limitations

- The model has a limited vocabulary of 97 tokens, which may affect its performance on certain tasks or domains.
- The maximum sequence length is 2048 tokens, which may not be sufficient for very long text generation tasks.


## Acknowledgements

 - Thanks to Andrej Karpathy for his excellent videos on how to train GPT from scratch
 - Sir Arthur Conan Doyle for the amazing stories :)