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

Install dependencies:
```bash
pip install transformers torch soundfile datasets
pip install git+https://github.com/huggingface/transformers.git

