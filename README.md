# AlignGuard-LoRA

# Usage Steps

1. Install dependencies:

```bash
pip install torch transformers peft scikit-learn scipy pandas
```

2. (Optional) Set your Hugging Face token for private models:

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
```

3. Prepare two CSV files: one for safe examples and one for unsafe examples. Each CSV must have the columns `instruction` and `response`.

4. Run training from the package CLI:

```bash
python -m alignguard.main \
	--model_path <PRETRAINED_MODEL> \
	--safe_csv_path safe.csv \
	--unsafe_csv_path unsafe.csv \
	--output_dir ./alignguard_output
```

5. Useful CLI options:

```text
--num_epochs, --batch_size, --learning_rate
--enable_poison_detection (flag)
--test_prompts <file>  # run inference using prompts from a text file
--inference_only        # skip training and only run inference on a saved adapter
```

6. Programmatic usage (minimal):

```python
from transformers import AutoTokenizer
from alignguard import AlignGuardConfig, AlignGuardModel, load_data_from_csvs, SafetyDataset, run_inference

texts, labels, safety_labels = load_data_from_csvs('safe.csv', 'unsafe.csv')
tokenizer = AutoTokenizer.from_pretrained('<PRETRAINED_MODEL>')
config = AlignGuardConfig()
model = AlignGuardModel('<PRETRAINED_MODEL>', config, tokenizer)
dataset = SafetyDataset(texts, labels, safety_labels, tokenizer)
model.train(train_dataset=dataset, output_dir='./alignguard_output')

# Run inference on the saved adapter or directly:
run_inference('<PRETRAINED_MODEL>', './alignguard_output', 'test_prompts.txt')
```

7. Check `./alignguard_output` for checkpoints and inference results.
