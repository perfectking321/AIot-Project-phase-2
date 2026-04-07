🔍 GitHub Projects for GUI Agent Grounding & Datasets

  Based on my search, here are the most relevant projects for your use case:

  🌟 Top GUI Agent Models (Ready to Use)

  Stars: 1,766
  Project: https://github.com/showlab/ShowUI
  Description: CVPR 2025 - End-to-end Vision-Language-Action model for GUI Agent

    & Computer Use. Best fit for your needs!
  ────────────────────────────────────────
  Stars: 1,167
  Project: https://github.com/zai-org/CogAgent
  Description: Open-source end-to-end VLM-based GUI Agent
  ────────────────────────────────────────
  Stars: 972
  Project: https://github.com/OSU-NLP-Group/Mind2Web
  Description: NeurIPS'23 - First LLM-based web agent benchmark with dataset
  ────────────────────────────────────────
  Stars: 585
  Project: https://github.com/niuzaisheng/ScreenAgent
  Description: Computer Control Agent driven by VLM (IJCAI-24)
  ────────────────────────────────────────
  Stars: 477
  Project: https://github.com/njucckevin/SeeClick
  Description: Visual GUI Agent with model, data, and code

  📊 Datasets for Training

  Stars: 972
  Project: https://github.com/OSU-NLP-Group/Mind2Web
  Description: 2,000+ web tasks with human demonstrations
  ────────────────────────────────────────
  Stars: 356
  Project: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
  Description: GUI Grounding for high-resolution computer use
  ────────────────────────────────────────
  Stars: 142
  Project: https://github.com/google-research-datasets/screen_qa
  Description: 86K Q&A pairs over mobile screenshots (Google)
  ────────────────────────────────────────
  Stars: 6
  Project: https://github.com/google-research-datasets/mug
  Description: User-agent interaction data (language + clicks)

  🏆 Benchmarks

  Stars: 2,741
  Project: https://github.com/xlang-ai/OSWorld
  Description: NeurIPS 2024 - Benchmark for multimodal agents in real computer
    environments
  ────────────────────────────────────────
  Stars: 1,064
  Project: https://github.com/MinorJerry/WebVoyager
  Description: End-to-end web agent with multimodal models

  ---
  🎯 My Recommendation for Your VOXCODE Project

  Option 1: Use ShowUI (Best for your case)

  ShowUI is specifically designed for what you need:
  - Understands semantic meaning (e.g., "C drive" → "Local Disk (C:)")
  - Trained on GUI grounding tasks
  - Works with screenshots + natural language

  # Install ShowUI
  pip install showui  # or clone from GitHub

  Option 2: Use SeeClick for Element Grounding

  SeeClick is trained to click on the correct element given a natural language
  description:
  - "Click the video about dead curse" → clicks the video title, not the channel
   icon

  Option 3: Fine-tune on Mind2Web Dataset

  If you want to train your own model:
  - Mind2Web has 2,000+ real web tasks
  - Each task has: instruction, screen, correct action

  ---
  🔧 Quick Integration: Using ShowUI API

  ShowUI has a free API you're already partially using. Let me check if we can
  improve the integration:
