import evaluate
wer = evaluate.load("wer")
predictions = [...]  # list read from base_transcriptions.txt
references = [ex["text"] for ex in ds["test"]]
wer_score = wer.compute(predictions=predictions, references=references)
print("WER:", wer_score)

sum(p.numel() for p in model.parameters() if p.requires_grad)
