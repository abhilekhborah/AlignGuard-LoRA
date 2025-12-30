import os
import logging
import argparse

from alignguard import (
    AlignGuardConfig,
    set_seed,
    AlignGuardModel,
    SafetyDataset,
    load_data_from_csvs,
    run_inference,
)

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder for HF token (do NOT hardcode keys here) - set via environment if needed
HF_TOKEN = os.environ.get("HF_TOKEN", "<YOUR_HF_TOKEN>")


def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="AlignGuard fine-tuning with separate safe and unsafe CSV data")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the pretrained LLaMA model")
    parser.add_argument("--safe_csv_path", type=str, required=True, help="Path to the CSV file with safe instruction-response pairs")
    parser.add_argument("--unsafe_csv_path", type=str, required=True, help="Path to the CSV file with unsafe instruction-response pairs")
    parser.add_argument("--output_dir", type=str, default="./alignguard_output", help="Directory to save the fine-tuned model")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank parameter")
    parser.add_argument("--lambda_sc", type=float, default=0.1, help="Weight for silhouette coefficient regularization")
    parser.add_argument("--lambda_null", type=float, default=0.1, help="Weight for null-space projection")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--safety_threshold", type=float, default=0.95, help="Poison detection threshold")
    parser.add_argument("--enable_poison_detection", action="store_true", help="Enable poison detection (disabled by default)")
    parser.add_argument("--inference_only", action="store_true", help="Only run inference on a saved model without training")
    parser.add_argument("--test_prompts", type=str, default=None, help="Path to a text file with test prompts for inference (one per line)")

    args = parser.parse_args()

    # If inference only, skip training
    if args.inference_only:
        if not args.output_dir or not os.path.exists(args.output_dir):
            logger.error("Must provide valid --output_dir for inference_only mode")
            return

        logger.info(f"Running inference only on saved model at: {args.output_dir}")
        run_inference(args.model_path, args.output_dir, args.test_prompts)
        return

    logger.info(f"Alignment preservation settings:")
    logger.info(f"  - Silhouette Coef. Weight (λSC): {args.lambda_sc}")
    logger.info(f"  - Null-Space Proj. Weight (λnull): {args.lambda_null}")
    logger.info(f"  - Poison Detection: {'Enabled' if args.enable_poison_detection else 'Disabled'}")
    if args.enable_poison_detection:
        logger.info(f"  - Poison Detection Threshold: {args.safety_threshold}")

    config = AlignGuardConfig(
        lambda_sc=args.lambda_sc,
        lambda_null=args.lambda_null,
        poison_threshold=args.safety_threshold,
        lora_r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        enable_poison_detection=args.enable_poison_detection,
    )

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        logger.warning("Tokenizer could not be loaded at this time; will try later when initializing the model.")

    if tokenizer and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info(f"Added pad token. Tokenizer size: {len(tokenizer)}")

    logger.info(f"Initializing AlignGuard for alignment-preserved fine-tuning...")
    safe_lora = AlignGuardModel(args.model_path, config, tokenizer)

    texts, labels, safety_labels = load_data_from_csvs(args.safe_csv_path, args.unsafe_csv_path)

    dataset = SafetyDataset(texts, labels, safety_labels, safe_lora.tokenizer, max_length=args.max_length)

    train_dataset = dataset
    eval_dataset = None

    logger.info(f"Starting fine-tuning with alignment preservation...")
    safe_lora.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if args.output_dir:
        logger.info(f"Training complete. Model saved to {args.output_dir}")
        run_inference(args.model_path, args.output_dir, args.test_prompts)


if __name__ == "__main__":
    main()
