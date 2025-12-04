from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

processor = Wav2Vec2Processor.from_pretrained(model_name)
base_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
# evaluate on ds["test"]
with torch.no_grad():
    for i, ex in enumerate(ds["test"]):
        input_values = torch.tensor(ex["input_values"]).unsqueeze(0).to(device)
        logits = base_model(input_values).logits.cpu().numpy()
        pred_ids = np.argmax(logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        out_lines.append(pred_str)

# write file
with open("base_transcriptions.txt","w",encoding="utf-8") as f:
    for l in out_lines:
        f.write(l+"\n")
