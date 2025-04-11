# Text2Speech-T5
A powerful text-to-speech tool using Microsoft's SpeechT5 model with speaker voice customization, built on PyTorch and Hugging Face Transformers.

# 🗣️ SpeakMyText: Realistic Text to Speech with SpeechT5

This repository showcases a Python-based Text-to-Speech (TTS) tool powered by Microsoft's `SpeechT5` model. It allows users to input any text (up to 500 characters) and generate human-like speech, with options to select different speaker voices using real x-vector embeddings.

## 🚀 Features

- 🔊 Convert any text into lifelike speech
- 🧠 Powered by Hugging Face `SpeechT5` and HiFi-GAN vocoder
- 🧬 Choose different speaker identities (e.g., male/female)
- ✂️ Supports long input text by breaking into manageable chunks
- 📦 Easy to run in Jupyter or Colab

  👨‍💻 Author
Riaz Khan — @riazhatvi

⭐ Show Some Love
If you found this useful, please consider giving it a ⭐ on GitHub!
Follow me for Latest AI Updates: https://www.youtube.com/c/RiazHatvi

## 🧰 Requirements

```bash
pip install transformers torch soundfile datasets
pip install git+https://github.com/huggingface/transformers.git

```
🧪 How to Use
```
Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/Text2Speech-T5.git
cd Text2Speech-T5
```
Step 2: Run the script in a Jupyter Notebook or Colab
```bash
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import numpy as np
from datasets import load_dataset
from IPython.display import Audio

# Load models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

def get_custom_input():
    text = input("Enter your text (max 500 characters):\n")[:500]
    speaker_idx = int(input("Choose speaker index (e.g., 7306 for female, 0 for male): ") or "7306")
    return text, speaker_idx

def generate_custom_speech():
    text, speaker_idx = get_custom_input()
    try:
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker_idx]["xvector"]).unsqueeze(0)
    except:
        print("Invalid speaker index! Using default.")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    full_audio = []
    for chunk in chunks:
        inputs = processor(text=chunk, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        full_audio.append(speech.numpy())

    final_audio = np.concatenate(full_audio)
    display(Audio(final_audio, rate=16000))
```
# Run it
generate_custom_speech()
📊 Speaker Embeddings
We use Matthijs/cmu-arctic-xvectors dataset for voice style control. Try different indices to switch voices.
Install dependencies:



