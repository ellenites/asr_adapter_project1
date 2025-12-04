from datasets import load_dataset, Audio
ds = load_dataset("DigitalUmuganda/ASR_Fellowship_Challenge_Dataset")

if "audio_filepath" in ds["train"].column_names:
    def remap(example):
        example["audio"] = example["audio_filepath"]
        return example
    ds = ds.map(remap)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
