from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="experiments/adapter_run_001",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    learning_rate=3e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=ds["train"], eval_dataset=ds["validation"],
                  data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()

adapter_state = {f"layer_{i}": layer.adapter.state_dict() for i, layer in enumerate(model.wav2vec2.encoder.layers)}
torch.save(adapter_state, "experiments/adapter_run_001/adapters_only.pth")
