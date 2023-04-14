"""Train a model for chatbot."""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    T5Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .utilities.config import Config


async def train(config: Config = Config()) -> None:
    """Train a model for chatbot."""

    logger = config.get_logger("Train")

    if torch.cuda.is_available():
        logger.info("Initializing CUDA...")
        torch.cuda.init()
        logger.info("CUDA initialized")

    logger.info("Loading base model...")
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(config.get("train.base_model"))
    logger.info("Base model loaded")

    logger.info("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(config.get("train.base_model", required=True))
    tokenizer.do_lower_case = True
    logger.info("Tokenizer loaded")

    logger.info("Adding special tokens...")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<user>", "<ai>", "<req>", "<res>"]})
    logger.info(tokenizer.special_tokens_map)
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Special tokens added")

    logger.info("Generating training dataset...")

    dataset = []
    with open(config.get("train.csv_file", required=True), "r", encoding="utf-8") as src:
        logger.info("Loading training data: (header=%s)", next(src))
        for i, line in enumerate(src, 1):
            if not line:
                continue
            try:
                prompt, completion = [
                    col.strip("「」\n\t 、。,.").replace("「", "『").replace("」", "』") for col in line.split(",")
                ]
                if not prompt or not completion:
                    continue
                dataset.append((f"<user>{prompt}", f"<ai>{completion}"))
            except Exception as err:
                logger.fatal("Failed to parse line (%d): %s", i + 1, line)
                logger.fatal(err)
                raise

    if config.get("train.generate_pairs", True):
        logger.info("Generating pairs...")
        for i in range(len(dataset) - 1):
            left_prompt, left_completion = dataset[i]
            right_prompt, right_completion = dataset[i + 1]
            dataset.append((left_prompt + left_completion + right_prompt, right_completion))

    logger.debug("Dataset: %s", dataset)

    train_dataset: Dataset = Dataset.from_dict(
        tokenizer.batch_encode_plus(dataset, padding=True, truncation=True, return_tensors="pt")
    )

    logger.info("Training dataset generated: %s", train_dataset)

    logger.info("Initializing trainer...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, **config.get("train.data_collator", {}))

    training_args = TrainingArguments(**config.get("train.training_args", {}))

    generation_config = config.get("chat_engine.generation_config", {})

    class Callback(TrainerCallback):
        """Callback to test the model after each epoch."""

        def on_epoch_end(self, *args, **kwargs):
            """Test the model after each epoch."""

            for text in config.get("train.test_inputs", []):
                input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                output_ids = model.generate(input_ids, **generation_config)
                logger.info(tokenizer.decode(output_ids[0], skip_special_tokens=False))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,  # type: ignore
        callbacks=[Callback()],
    )
    logger.info("Trainer initialized")

    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(config.get("train.training_args.output_dir"))
    logger.info("Tokenizer saved")

    logger.info("Training...")
    trainer.train()
    logger.info("Training completed")

    logger.info("Saving models...")
    trainer.save_state()
    trainer.save_model()
    logger.info("Model saved")

    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(config.get("train.training_args.output_dir"))
    logger.info("Tokenizer saved")
