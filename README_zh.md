<h2 align="center">VoxCPM2：基于连续表征的多语言语音合成、创意音色设计与高保真声音克隆</h2>

<p align="center">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

<p align="center">
  <a href="https://github.com/OpenBMB/VoxCPM/"><img src="https://img.shields.io/badge/Project%20Page-GitHub-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo"><img src="https://img.shields.io/badge/Live%20Playground-Demo-orange" alt="Live Playground"></a>
  <a href="https://voxcpm.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/Docs-ReadTheDocs-8CA1AF" alt="Documentation"></a>
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
  👋 欢迎加入社区，参与讨论与交流！
  <br>
  <a href="./assets/feishu-group.png" style="display:inline-block;vertical-align:middle; margin-left: 10px;">
    <img src="./assets/feishu-logo.png" width="16" height="16" style="vertical-align:middle;"> 飞书群
  </a>
  &nbsp;|&nbsp;
  <a href="https://discord.gg/KZUx7tVNwz" style="display:inline-block;vertical-align:middle;">
    <img src="./assets/discord-logo.png" width="16" height="16" style="vertical-align:middle;"> Discord
  </a>
</p>

VoxCPM 是一个**无离散音频分词器**（Tokenizer-Free）的语音合成系统，通过端到端的**扩散自回归架构**直接生成连续语音表征，绕过对音频的离散编码步骤，实现高度自然且富有表现力的语音合成。

**VoxCPM2** 是最新的版本 — 基于 [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) 基座构建，总计 **20亿** 参数，在超过 **200万小时** 的多语种音频数据上训练，支持 **30种全球语言+9种中文方言**、**音色设计**、**可控声音克隆**，原生输出 **48kHz** 高质量音频。

### ✨ 核心特性

- 🌍 **30种语言语音合成** — 直接输入原始文本即可合成（支持语言详见下文），无需额外语言标签
- 🎨 **音色设计** — 用自然语言描述（性别、年龄、音色、情绪、语速……）凭空创建全新音色，无需参考音频
- 🎛️ **可控声音克隆** — 从参考音频片段克隆任意声音，可叠加风格指令控制情绪、语速和表现力，同时保持原始音色
- 🎙️ **极致克隆** — 提供参考音频及其文本内容，模型接着参考音频进行无缝续写，从而精准还原声音细节特征（与 VoxCPM1.5 一致）
- 🔊 **48kHz 高质量音频** — 输入 16kHz 参考音频，通过 AudioVAE V2 的非对称编解码设计直接输出 48kHz 高质量音频，内置超分能力
- 🧠 **语境感知合成** — 根据文本内容自动推断合适的韵律和表现力
- ⚡ **实时流式合成** — 在 NVIDIA RTX 4090 上 RTF 低至 ~0.3，通过 [Nano-VLLM](https://github.com/a710128/nanovllm-voxcpm) 加速后可达 ~0.13
- 📜 **完全开源，商用就绪** — 权重和代码基于 [Apache-2.0](LICENSE) 协议发布，免费商用

<summary><b>🌍 支持的语言（30种）</b></summary>
<br>
阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印尼语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、菲律宾语、泰语、土耳其语、越南语

中国方言：四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话


### 最新动态

* **[2025.07]** 📝 个人注：正在学习极致克隆功能，尝试用于中文播客制作，效果相当不错
* **[2026.04]** 🔥 发布 **VoxCPM2** — 20亿参数，30种语言，音色设计与可控声音克隆，48kHz 高质量音频
