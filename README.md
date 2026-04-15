<h2 align="center">VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning</h2>

<p align="center">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/OpenBMB/VoxCPM/"><img src="https://img.shields.io/badge/Project%20Page-GitHub-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo"><img src="https://img.shields.io/badge/Live%20Playground-Demo-orange" alt="Live Playground"></a>
  <a href="https://voxcpm.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Docs-ReadTheDocs-8CA1AF" alt="Documentation"></a>
  <a href="https://huggingface.co/openbmb/VoxCPM2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VoxCPM2-yellow" alt="Hugging Face"></a>
  <a href="https://modelscope.cn/models/OpenBMB/VoxCPM2"><img src="https://img.shields.io/badge/ModelScope-VoxCPM2-purple" alt="ModelScope"></a>
  <a href="https://openbmb.github.io/voxcpm2-demopage/"><img src="https://img.shields.io/badge/DemoPage-Audio Samples-red"></a>
  
</p>

<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="35%">
  <br><br>
  <a href="https://trendshift.io/repositories/17704" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17704" alt="OpenBMB%2FVoxCPM | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<br>

<p align="center">
  👋 Join our community for discussion and support!
  <br>
  <a href="./assets/feishu-group.png" style="display:inline-block;vertical-align:middle; margin-left: 10px;">
    <img src="./assets/feishu-logo.png" width="16" height="16" style="vertical-align:middle;"> Feishu
  </a>
  &nbsp;|&nbsp;
  <a href="https://discord.gg/KZUx7tVNwz" style="display:inline-block;vertical-align:middle;">
    <img src="./assets/discord-logo.png" width="16" height="16" style="vertical-align:middle;"> Discord
  </a>
</p>

> **Personal fork note:** I'm using this project to experiment with voice cloning for language learning demos. Main branch tracks upstream; my experiments live in the `experiments/` branch.
>
> **My tested languages:** Japanese and Spanish cloning work well with ~5s reference audio. For Mandarin tones, I found 10s+ reference clips give noticeably better results. See `experiments/lang_notes.md` for details.

VoxCPM is a **tokenizer-free** Text-to-Speech system that directly generates continuous speech representations via an end-to-end **diffusion autoregressive architecture**, bypassing discrete tokenization to achieve highly natural and expressive synthesis.

**VoxCPM2** is the latest major release — a **2B** parameter model trained on **over 2 million hours** of multilingual speech data, now supporting **30 languages**, **Voice Design**, **Controllable Voice Cloning**, and **48kHz** studio-quality audio output. Built on a [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) backbone.

### ✨ Highlights

- 🌍 **30-Language Multilingual** — Input text in any of the 30 supported languages and synthesize directly, no language tag needed
- 🎨 **Voice Design** — Create a brand-new voice from a natural-language description alone (g
