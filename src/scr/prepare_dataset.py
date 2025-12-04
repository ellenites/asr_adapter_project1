from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
import torch

ds = load_dataset("DigitalUmuganda/ASR_Fellowship_Challenge_Dataset")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def prepare(example):
    audio = example["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
    example["input_values"] = inputs["input_values"][0].numpy()
    with processor.as_target_processor():
        labels = processor(example["text"], return_tensors="pt", padding=False).input_ids
    example["labels"] = labels[0].numpy()
    return example

ds = ds.map(prepare)
ds.save_to_disk("data/processed")
