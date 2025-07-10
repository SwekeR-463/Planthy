# Planthy -- Plant Health Monitor Agent

## Description

A multimodal AI agent that diagnoses plant health issues by analyzing images and text queries, providing actionable recommendations. It leverages real-time web search and advanced reasoning to assist gardeners and farmers, deployed as a user-friendly web app on [Hugging Face Spaces](https://huggingface.co/spaces/Agents-MCP-Hackathon/Planthy).

<img width="1920" height="1080" alt="Screenshot from 2025-06-05 18-30-15" src="https://github.com/user-attachments/assets/2ad1e24e-15e3-43a3-9e40-7705d81536eb" />

## Tech Stack

* LlamaIndex: ReAct agent and tool orchestration.

* Gemini: Multimodal image and text processing (`gemini-2.0-flash`).

* DuckDuckGo Search: Real-time plant health information retrieval.

* Gradio: Web interface for user interaction.

## Functionality

* Image Analysis: Upload a plant image to identify type, condition, symptoms, and confidence.

* Text Queries: Input queries (e.g., "What’s wrong with my plant?") to get tailored recommendations.

* ReAct Reasoning: Combines image analysis and web search for intelligent, context-aware responses.

* Web Interface: Gradio UI for easy image uploads, query input, and Markdown-formatted outputs (diagnosis, recommendations, reasoning).

* Real-Time Data: Fetches plant care tips via DuckDuckGo search, eliminating the need for a local dataset.

## Demo

[Video](https://drive.google.com/file/d/1swZV-Eua6dt0QTZB8uH6cq4evhd_1dHZ/view?usp=sharing)
