# AutoMat
AutoMat focuses on characterization to property analysis and is preparing for NeurIPS 2025.

# demo
┌───────────┐   upload      ┌─────────────────┐   json / npy      ┌───────────────┐
│  前端 UI  │ ───────────► │   API Gateway   │ ───────────────► │  STEM 模型微服 │
│(Streamlit │              │ (FastAPI)       │                  │(GPU 推理+REST) │
│/React)    │ ◄─────────── │                 │ ◄─────────────── │               │
└───────────┘  websocket    └─────────────────┘    callback      └───────────────┘
     ▲                                                         result.json
     │Markdown + 图表                         ┌─────────────────────┐
     │                                         │  LLM Orchestrator  │
     │                                         │   (LangChain)      │
     │           answer / plot (LLM) ◄─────────┤  统一 LLM Driver   │
     │                                         │  GPT-4o / Qwen /   │
     │                                         │  Llama / DeepSeek  │
     │ STEM image (可选) ─────┐                └─────────────────────┘
     └────────────────────────┴─►  GPT-4o Vision (可选)
