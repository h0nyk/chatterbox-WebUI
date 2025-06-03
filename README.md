<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS - Windows WebUI Fork

> **🚀 This is a fork of the original [Resemble AI Chatterbox TTS](https://github.com/resemble-ai/chatterbox) with enhanced features for Windows users!**

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/XqS7RxUp)

## 🆕 What's New in This Fork

### **🎯 One-Click Windows Installation**
- **Automatic environment setup** with CUDA 11.8 support
- **No technical knowledge required** - just run `install.bat`
- **Instant launcher** - double-click `run_tts.bat` to start

### **🌐 Enhanced Gradio WebUI**
- **Professional web interface** accessible at `http://127.0.0.1:7860/`
- **Voice cloning capabilities** with reference audio upload
- **Real-time parameter adjustment** (exaggeration, temperature, CFG/pace)
- **Example prompts** and intuitive controls

### **🚀 Smart Long-Form Processing** *(New Feature!)*
What the original didn't have:
- **Smart Text Chunking**: Automatically splits long documents at sentence boundaries
- **Parallel Processing**: Processes multiple chunks simultaneously for 4x faster generation
- **Seamless Audio Stitching**: Combines chunks into one cohesive audio file
- **Progress Tracking**: Real-time progress indicators during generation
- **Voice Consistency**: Maintains the same cloned voice across all chunks
- **Configurable Batch Size**: Adjust parallel processing for your hardware

**Perfect for**: Articles, books, scripts, documentation, or any text longer than 300 characters!

## 🚀 Quick Start (Windows)

### Installation (30 seconds)
```bash
git clone https://github.com/Saganaki22/chatterbox-WebUI
cd chatterbox-WebUI
install.bat
```

### Launch WebUI
```bash
run_tts.bat
```

Then open: **http://127.0.0.1:7860/**

That's it! 🎉

---

## 📖 About Chatterbox TTS

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce Chatterbox, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out.

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

## 🔥 Key Features
- **SoTA zeroshot TTS** - State-of-the-art voice cloning
- **0.5B Llama backbone** - Powerful language model foundation
- **Unique exaggeration/intensity control** - First open-source TTS with emotion control
- **Ultra-stable with alignment-informed inference** - Consistent, high-quality output
- **Trained on 0.5M hours** of cleaned data
- **Watermarked outputs** - Built-in Perth watermarking for responsible AI
- **Easy voice conversion** - Simple reference audio upload
- **[Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)** in side-by-side comparisons

## 💡 Usage Tips

### **General Use (TTS and Voice Agents):**
- The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts.
- If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

### **Expressive or Dramatic Speech:**
- Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
- Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.

### **Long-Form Content:**
- Use the **Long Form Content** tab for documents, articles, or scripts
- Adjust **Batch Size** (1-8) based on your GPU memory
- Set **Chunk Size** (100-500 chars) for optimal sentence splitting
- Upload reference audio once - it applies to all chunks automatically

## 🔧 Manual Installation (Advanced Users)

If you prefer manual installation or need to customize:

```bash
# Create virtual environment
python -m venv myenv
myenv\Scripts\activate.bat

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ChatterboxTTS and Gradio
pip install chatterbox-tts gradio

# Run the WebUI
python gradio_tts_app.py
```

## 🐍 Python API Usage
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# Voice cloning with reference audio
AUDIO_PROMPT_PATH="YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```

## 🛠️ WebUI Features

### **Single Text Tab**
- Quick TTS generation for short texts
- Voice cloning with reference audio upload
- Real-time parameter adjustment
- Example prompts included

### **Long Form Content Tab** *(Fork Enhancement)*
- Process documents, articles, books
- Smart sentence-boundary chunking
- Parallel processing for speed
- Progress tracking with real-time updates
- Automatic audio stitching
- Voice consistency across all chunks

### **Utilities Tab**
- File cleanup and management
- Tips for best results
- Parameter explanations

## 🏗️ Original Repository

This fork is based on the original [Resemble AI Chatterbox TTS](https://github.com/resemble-ai/chatterbox). All core TTS functionality and model weights remain unchanged - we've simply added Windows-friendly installation and an enhanced web interface with long-form processing capabilities.

## 🙏 Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

## 🔒 Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

## 💬 Official Discord

👋 Join us on [Discord](https://discord.gg/XqS7RxUp) and let's build something awesome together!

## ⚠️ Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
