# LLMOps.Space

LLMOps.Space is a global community for LLM practitioners.💡📚 

## Get Started

Start with joining the - [LLMOps Discord server](https://llmops.space/discord) and introducing yourself.

## Table of Contents
- [About LLMOps.Space](#about-llmopsspace)
- [LLMOps Companies & Products](#llmops-companies--products)
- [Upcoming Talks & Demos (Ours)](#upcoming-talks--demosours)
- [Educational Materials](#educational-materials)
- [Propose a Talk ](#propose-a-talk)
- [List of LLM Consultants](#list-of-llm-consultants)
- [LLM Modules](#)
- [List of LLM-Related Events](#List-of-LLM-Related-Events)
- [Beta Program Listings](Beta-Program-Listings)
- [Code of Conduct](#Code-of-Conduct)

## About LLMOPs.Space

💡LLMOps space is a global community for LLM practitioners. The community will focus on content, discussions, and events around topics related to deploying LLMs into production. But with one special emphasis - content, lists & announcements are supposed to be standardized, organized, and focused on the needs of community members. <br>

💬 The core of the community revolves around our - [Discord](llmops.space/discord)

While this repo is more about preserving & organizing the knowledge shared there. 

🔍 Please read the community code of conduct before you join, we are pretty strict about it and will ⛔️ ban folks & companies pretty easily when violations occur.

🙋‍♂️ LLMOps.Space was initiated by a couple of us at - [Deepchecks](deepchecks.com) that felt we had a need for a community like this, and we’re currently looking for a group of volunteers to help maintain the community. Please reach out to an admin if you’re interested.

🕝 We currently aren’t signing on companies/vendors as sponsors, the main need is for more hands-on help at this point. This may change over time as the community culture stabilized, and in the meantime - we appreciate your patience. 


## LLMOps Companies & Products

We’ve curated an initial mapping of the companies and products providing value in LLMOps while associating each offering with only one category (to keep the navigation simple). 

<br>

![LLMOps Space](https://deepchecks.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0585b05d-b941-4257-b5a2-4929959f926c%2FFrame_1_(6).png?table=block&id=822dcd77-51a8-4599-9032-96a9928bfea1&spaceId=97e50776-4bd0-4e81-9b7b-750c81676752&width=2000&userId=&cache=v2)

<br>

### Data Labelling 
- [Toloka](https://toloka.ai/large-language-models/) - Get human input at all stages of LLM development: Pre-training, fine-tuning and RLHF
- [Labelbox](https://labelbox.com/solutions/large-language-models/) - Customizable data engine built to produce high-quality training data to help AI teams build ML models faster
- [Argilla](https://argilla.io/) - Open-source data curation platform for LLMs using human and machine feedback loops
- [Surge](https://www.surgehq.ai/rlhf) - Provide NLP datasets to train LLM and AI
- [Scale](https://scale.com/rlhf) - Data provider to train and validate models. specifically use RLHF to optimize LLM applications

### Data Storage & Management
- [LlamaIndex](https://www.llamaindex.ai/) - LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models.
- [Activeloop Deep Lake](https://www.activeloop.ai/) - Deep Lake combines the power of both Data Lakes & Vector Databases to build, fine-tune, & deploy enterprise-grade LLM solutions, & iteratively improve them over time.
- [DAGSHub](https://dagshub.com/use-cases/llm/) - Managing data and labels for RLHF. Loggings prompts and fine tuned LLMs for deployment
- [Weaviate](weaviate.io) - Weaviate is an open-source vector database


### End-to-end LLM Platform
- [Dataloop](https://dataloop.ai/blog/introducing-dataloops-rlhf-studio-revolutionizing-reinforcement-learning-with-human-feedback/) - Data management cycle, from data labeling, automating data ops, deploying production pipelines, and weaving the human-in-the-loop
- [ClearML](https://clear.ml) - The First Generative AI Platform That Transcends Enterprise ChatGPT Challenges
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview) - Allows developers to use LLMs for enterprise-grade applications via summarization, conversational AI, Writing Assistance, Knowledge Mining, SW development.
- [GCP Palm API](https://developers.generativeai.google/products/palm) - The PaLM API is based on Google’s next generation model, PaLM 2, which excels at a variety of capabilities. 
- [AWS Kendra](https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/) - Intelligent enterprise search service that helps you search across different content repositories with built-in connectors.
- [Databricks Lakehouse Platform](https://www.databricks.com/product/machine-learning/large-language-models) - Allows enterprises to access LLMs to integrate into workflows as well as platform capabilities to fine-tune LLMs using own company data for better domain performance.
- [NVIDIA NeMo Megatron](https://www.nvidia.com/en-in/deep-learning-ai/solutions/large-language-models/) - end-to-end enterprise framework that provides workflow for training with 3D parallelism techniques and is optimized for at-scale inference of large-scale models for language and image applications
- [Hugging Face](https://huggingface.co/) - Build, train and deploy state of the art models powered by the reference open source in machine learning.
- [Qwak](www.qwak.com) - Qwak is a fully managed, accessible, and reliable MLOps/LLMOps Platform. It allows builders to transform and store data, build, train, and deploy models, and monitor the entire Machine Learning pipeline. 


### Experiment Tracking
- [Databricks Mlflow](https://mlflow.org/docs/latest/llm-tracking.html) - Open source platform for managing the end-to-end machine learning lifecycle
- [Weights & Biases (W&B)](https://wandb.ai/site/solutions/llmops) - Platform to train, track, tune, and manage end-to-end LLM operations
- [TruLens (by TruEra)](https://www.trulens.org/) - Uses feedback functions to measure quality and effectiveness of your LLM application

### LLM API

- [AI21 Studio](https://www.ai21.com/studio) - API tool that generates text completions for an input prompt, question answering, and text classification
- [Anthropic](https://www.anthropic.com/product) - AI assitant that can process a lot of text, have natural conversations, get answers, automate workflows
- [OpenAI GPT-4](https://openai.com/gpt-4) - GPT-4 is a LLM (accepting image and text inputs, emitting text outputs) that, while less capable than humans in many real-world scenarios, exhibits human-level performance on various professional and academic benchmarks

### Low Code/Simplified LLM App Builder
- [Fixie AI](https://fixie.ai/) - Cloud-based platform-as-a-service that allows developers to build smart agents that couple LLMs with back-end logic to interface to data, systems, and tools.
- [One AI](https://www.oneai.com/) - Select from our library, fine-tune, or build your own capabilities to analyze and process text, audio and video at scale.

### Model Training & Fine-Tuning 
- [Cerebras AI Model Studio Launchpad](https://www.cerebras.net/blog/fine-tuning-with-cerebras-ai-model-studio) - Resource for fine-tuning models using domain specific datasets and tasks
- [MosaicML](https://www.mosaicml.com/) - Developer of software infrastructure and artificial intelligence training algorithms designed to improve the efficiency of neural networks.

### Monitoring, Testing, or Validation
- [Deepchecks](https://deepchecks.com/) - Deepchecks NLP is a holistic tool for testing, validating and monitoring your NLP models and data, throughout the model’s lifecycle. Open souce supports classification & token classification, beta program supports GPT4 and similar LLMs.
- [Arize](https://arize.com/llm/) - Evaluate LLM responses, pinpoint where to improve with prompt engineering, and identify fine-tuning opportunities using vector similarity search.
- [Mona Labs](https://www.monalabs.io/) - GPT Monitoring by Mona

### Orchestration & Model Deployment
- [Dstack AI](https://dstack.ai/) - Cost-effective LLM development
- [Run:AI](https://pages.run.ai/hubfs/PDFs/Enabling%20LLM%20Adoption%20at%20the%20Enterprise%20-%20Datasheet.pdf) - Platform for end-to-end LLM lifecycle management, enabling enterprises to fine-tune, prompt engineer, and deploy LLM models with ease
- [Iguazio- acquird by McKinsey](https://www.iguazio.com/sessions/mlops-for-llms/) - MLOps acceleration with real-time serving pipeline, monitoring and re-training, and integrated CI/CD functionalities
- [Anyscale](https://www.anyscale.com/large-language-models) - Open source, free, cloud-based LLM-serving infrastructure designed to help developers choose and deploy the right technologies and approach for their LLM-based applications. 
- [ZenML](https://zenml.io/home) - Run MLOps workflows on any infrastructure with ZenML.


### Prompt Engineering & Management
- [Langchain](https://github.com/hwchase17/langchain) - Developer of a language model framework designed to power applications that integrate with other sources of data and interact with their environment.
- [PromptLayer](https://promptlayer.com/) - Devtool that allows you to track, manage, and share your GPT prompt engineering. It acts as a middleware between your code and OpenAI's python library, recording all your API requests and saving relevant metadata for easy exploration and search in the PromptLayer dashboard.
- [Comet](https://www.comet.com/site/products/llmops/) - Use Prompt Management and query models in Comet to iterate quicker, identify performance bottlenecks, and visualize the internal state of the Prompt Chains
- [Parea](https://www.parea.ai/) - End-to-end platform that offers automatic prompt optimization, version control, prompt comparisons and sharing
- [Dust.tt](https://dust.tt/) - Easy to use graphical UI to build chains of prompts, as well as a set of standard blocks and a custom programming language to parse and process language model outputs.
- [OpenPrompt](https://thunlp.github.io/OpenPrompt/) - OpenPrompt is a library built upon PyTorch and provides a standard, flexible and extensible framework to deploy the prompt-learning pipeline.
- [Orquesta](https://orquesta.cloud/platform/ai-llm-prompts) - Centralize your prompts in a single source of truth, experiment with them on multiple LLMs for quality and pricing, customize them for specific contexts, and collect feedback on accuracy and economics.
- [Runbear, Inc.](https://langbear.runbear.io/introduction) - Prompt management & A/B testing platform


### Security, Privacy & Compliance
- [Cyera SafeType](https://www.cyera.io/safetype) - SafeType anonymizes sensitive data typed into ChatGPT to avoid misuse and accidental disclosures
- [Nightfall](https://docs.nightfall.ai/docs/content-filtering-sensitive-data-chatgpt) - Content filtering with ChatGPT to prevent exposure of sensitive customer and company data.

### Vector Search
- [Pinecone](https://www.pinecone.io/) - Vector Database
- [Elastic](https://www.elastic.co/what-is/vector-search) - Vector Search Engine
- [Searchium AI](https://www.searchium.ai/) - Vector search accelerator
- [Vespa AI](https://vespa.ai/) - Vespa is a fully featured search engine and vector database. It supports vector search (ANN), lexical search, and search in structured data, all in the same query.
- [Vectara](https://vectara.com/ ) - Vectara is LLM-powered search-as-a-service.
- [Jina AI](https://jina.ai/) - Jina lets you build multimodal AI services and pipelines that communicate via gRPC, HTTP and WebSockets, then scale them up and deploy to production.
- [Hyper Space](https://www.hyper-space.io/) - Real-time Hybrid Search Database.

### Vector Database
- [Milvis](https://milvus.io/) - Milvus is the world's most advanced open-source vector database, built for developing and maintaining AI applications.
- [Qdrant](https://qdrant.tech/) - Qdrant is a vector similarity search engine and vector database
- [Zilliz](https://zilliz.com/) - Zilliz vector database management system - powered by Milvus.

<br>

## Trending LLMs

- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - The current Alpaca model is fine-tuned from a 7B LLaMA model [1] on 52K instruction-following data generated by the techniques in the Self-Instruct [2] paper, with some modifications that we discuss in the next section. 
- [BELLE](https://github.com/LianjiaTech/BELLE) - BELLE is more concerned with how to build on the foundation of open-source pre-trained large language models to help everyone obtain their own high-performing, instruction-driven language model, thereby lowering the barriers to research and application of large language models, especially Chinese ones.
- [Bloom](https://bigscience.huggingface.co/blog/bloom) - The BLOOM model has been proposed with its various versions through the BigScience Workshop. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact. 
- [dolly](https://github.com/databrickslabs/dolly) - dolly-v2-12b is a 12 billion parameter causal language model created by Databricks that is derived from EleutherAI’s Pythia-12b and fine-tuned on a ~15K record instruction corpus generated by Databricks employees and released under a permissive license (CC-BY-SA)
- [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b-instruct) - Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by TII based on Falcon-40B and finetuned on a mixture of Baize. It is made available under the Apache 2.0 license.
- [FastChat](https://github.com/lm-sys/FastChat) - FastChat is an open platform for training, serving, and evaluating large language model based chatbots. The core features include:
- [Gorilla LLM](https://gorilla.cs.berkeley.edu/) - Gorilla is a LLM that can provide appropriate API calls. It is trained on three massive machine learning hub datasets: Torch Hub, TensorFlow Hub and HuggingFace. 
- [GLM-6B (ChatGLM)](https://github.com/THUDM/ChatGLM-6B) - ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters.
- [GLM-130B (ChatGLM)](https://github.com/THUDM/GLM-130B) - GLM-130B is an open bilingual (English & Chinese) bidirectional dense model with 130 billion parameters, pre-trained using the algorithm of General Language Model (GLM). It is designed to support inference tasks with the 130B parameters on a single A100 (40G * 8) or V100 (32G * 8) server.
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - GPT-NeoX-20B, a 20 billion parameter autoregressive language model trained on the Pile, whose weights will be made freely and openly available to the public through a permissive license.
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) - MPT-7B is a transformer trained from scratch on 1T tokens of text and code. It is open source, available for commercial use, and matches the quality of LLaMA-7B.
- [PaLM 2](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/) -  PaLM 2 is a state-of-the-art language model with improved multilingual, reasoning and coding capabilities.
- [StableLM](https://github.com/Stability-AI/StableLM) - Stability AI released a new open-source language model, StableLM. The Alpha version of the model is available in 3 billion and 7 billion parameters, with 15 billion to 65 billion parameter models to follow.
- [Starcoder](https://huggingface.co/bigcode/starcoder) - StarCoder is Large Language Models for Code (Code LLMs) trained on permissively licensed data from GitHub, including from 80+ programming languages, Git commits, GitHub issues, and Jupyter notebooks.

## Upcoming Talks & Demos(Ours)

📝 If you’d like to explore giving an LLMOps Talk, please see [👉this👈](https://llmops.space/propose-a-talk) page.
<br>

## Educational Materials

Here you will find a collection of resources that we have compiled to help you learn more about LLMs fundamentals and related topics. 🏅
<br>

### **Recommended Blog Post**
Blogs related to LLMs that you’d love to read 📄

| Title       | Published Date  |
| ----------- | --------------- |
| [Releasing Swift Transformers: Run On-Device LLMs in Apple Devices](https://huggingface.co/blog/swift-coreml-llm)   | Aug 8, 2023 |
| [LLMOps: Bridging the Gap Between LLLMs and MLOps](https://www.projectpro.io/article/llmops/895)  | Aug 8, 2023         |
| [Interpretability Creationism](https://thegradient.pub/interpretability-creationism/)   | July 11, 2023        |
| [A comprehensive guide to learning LLMs (Foundational Models)](https://www.linkedin.com/pulse/comprehensive-guide-learning-llms-foundational-models-yeddula/)   | June 14, 2023        |
| [Deploying Large NLP Models: Infrastructure Cost Optimization](https://neptune.ai/blog/nlp-models-infrastructure-cost-optimization)   | Jun 05, 2023        |
| [Introduction to Large Language Models](https://medium.com/the-llmops-brief/introduction-to-large-language-models-9ac028d34732)   | May 17, 2023        |
| [10 Exciting Project Ideas Using Large Language Models (LLMs) for Your Portfolio](https://towardsdatascience.com/10-exciting-project-ideas-using-large-language-models-llms-for-your-portfolio-970b7ab4cf9e)   | May 15, 2023      |
| [How Lakehouse powers LLM for Customer Service Analytics in Insurance](https://www.databricks.com/blog/how-lakehouse-powers-nlp-customer-service-analytics-insurance)   | May 12, 2023        |
| [Effortless Fine-Tuning of Large Language Models with Open-Source H2O LLM Studio](https://h2o.ai/blog/effortless-fine-tuning-of-large-language-models-with-open-source-h2o-llm-studio/)   | May 12, 2023 |
| [Enhancing Product Search with Large Language Models (LLMs)](https://www.databricks.com/blog/enhancing-product-search-large-language-models-llms.html)   | Apr 26, 2023        |
| [Building LLM Applications for Production](https://news.ycombinator.com/item?id=35565212)   | Apr 15, 2023        |
| [Unleashing the Power of GPT-3: Fine-Tuning for Superhero Descriptions](https://towardsdatascience.com/unleashing-the-power-of-gpt-how-to-fine-tune-your-model-da35c90766c4)   | Feb 18, 2023        |
| [Understanding Large Language Models -- A Transformative Reading List](https://sebastianraschka.com/blog/2023/llm-reading-list.html)   | Feb 07, 2023        |
| [Build a GitHub Support Bot with GPT3, LangChain, and Python](https://dagster.io/blog/chatgpt-langchain)   | Jan 09, 2023        |
| [Beyond Words: Large Language Models Expand AI’s Horizon](https://blogs.nvidia.com/blog/2022/10/10/llms-ai-horizon/)   | Oct 10, 2022        |
| [Choosing the right language model for your NLP use case](https://towardsdatascience.com/choosing-the-right-language-model-for-your-nlp-use-case-1288ef3c4929)   | Sep 26, 2022        |
| [Evolution of Large Language Models — BERT, GPT3, MUM, and PaLM](https://towardsdatascience.com/self-supervised-transformer-models-bert-gpt3-mum-and-paml-2b5e29ea0c26)   | Jul 06, 2022        |
| [Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)   | Apr 04, 2022        |
| [Large Language Models: A New Moore's Law?](https://huggingface.co/blog/large-language-models)   | Oct 26, 2021        |
| [Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)   | Apr 12, 2021        |



### **Recommended Videos**
Here, we've curated a fantastic selection of insightful videos to expand your knowledge about LLMs and LLMOps, paving the way for a quick start in this space. 💡

| Topic       | Published Date      |
| ----------- | ------------------- |
| [Huggingface Agents: Multimodal Transformers Agents](https://youtu.be/SCYMLHB7cfY)    | May 13, 2023        |
| [LLMOps: Deployment and Learning in Production](https://youtu.be/Fquj2u7ay40)   | May 11, 2023        |
| [Launch an LLM App in One Hour](https://youtu.be/twHxmU9OxDU)   | May 11, 2023        |
| [LangChain Crash Course: Build a AutoGPT app in 25 minutes](https://youtu.be/MlK6SIjcjE8)   | Apr 24, 2023        |
| [Reinforcement Learning from Human Feedback: Progress and Challenges](https://www.youtube.com/live/hhiLw5Q_UFg)   | Apr 20, 2023        |
| [LangChain for Gen AI and LLMs](https://www.youtube.com/playlist?list=PLIUOU7oqGTLieV9uTIFMm6_4PXg-hlN6F)   | Jan 25, 2023        |
| [Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY)   | Jan 17, 2023        |
| [Transformers United 2023: Introduction to Transformers](https://youtu.be/XfpMkf4rD6E)   | Jan 10, 2023        |
| [The Future of LLMs, Foundation & Generative Models](https://youtu.be/Rp3A5q9L_bg)   | Oct 24, 2022        |
| [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo)   | Sep 8, 2022        |
| [Bloom (Text Generation Large Language Model - LLM): Step-by-step implementation](https://youtu.be/HOiBaH9gAlU)   | Aug 14, 2022        |
| [The Narrated Transformer Language Model](https://www.youtube.com/watch?v=-QH8fRhqFHM)   | Oct 26, 2020        |


### **Courses for Basics and Fundamentals**

Courses that we recommend and that explain the building blocks of the LLM space:

**[Stanford CS324 - Large Language Models](https://stanford-cs324.github.io/winter2022/)**

- In this course, students will learn the fundamentals about the modeling, theory, ethics, and systems aspects of large language models, as well as gain hands-on experience working with them.

**[Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/)**

- In this course, learn how transformers work, and dive deep into the different kinds of transformers and how they're applied in different fields.

**[Full Stack LLM Bootcamp](https://www.youtube.com/watch?v=twHxmU9OxDU&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ)**

- Learn best practices and tools for building LLM-powered apps

**[LangChain for LLM Application Development- Andrew Ng](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)**

- In LangChain for LLM Application Development, you will gain essential skills in expanding the use cases and capabilities of language models in application development using the LangChain framework.

**[LangChain & Vector Databases in Production](https://learn.activeloop.ai/courses/langchain)**

- In this course, you will learn how to leverage LangChain, a robust framework for building applications with LLMs, and explore Deep Lake, a groundbreaking vector database for all AI data.


## Propose a Talk 

At [LLMOps.Space](llmops.space) we’re aiming to have various types of talks (some may be combined to a single session):

- **Lightning talk** - flexible format (10-15 minutes)
- **Regular talk** - should include code or demonstrations (30-45 minutes)
- **Panel discussion** - flexible format (40-60 minutes)
- **Product/repo unveiling** - requires showing a product/repo that hasn’t yet been announced elsewhere (10-15 minutes)
- **Beta feedback session** - for products/repos that have already been tried by at least a couple of users, but haven’t been announced yet (20-60 minutes). Requires enabling hands-on access for users to try. It’s customary to offer SWAG to at least a subset of the active participants

Feel free to propose a talk [👉here👈,](https://forms.gle/jgu2A8YCZHiTfEYG8) and we’ll have a moderator review it. Talks will be selected according to various considerations: Lessons learned from previous talks, feedback about the speaker or the topic from past events, making a coherent curriculum for the participants, and more. 

Due to the many tasks managed by the small (and part-time) moderation team, we won’t be sending rejection notices. So if you haven’t received an acceptance message within 2 weeks, you can assume the talk wasn’t selected at this time. 

Thank you for your understanding. 🙏



## List of LLM Consultants

📝 If you’re a consultant with a specialty in LLMs and would want to like to enter our list, please fill out [👉this👈](https://forms.gle/ge76J1S3WYkS4Esa6) form.

🤓 If you’re looking for consultants, please check the below table! Rest assured, we're continuously vetting and adding new talented consultants to this list.

| Name               | LinkedIn            | LLMOps Link | Hourly Price Range |
| -------------------| ------------------- | ----------- | ------------------ |
| Daniel Tannor |https://www.linkedin.com/in/dtannor/ | https://llmops.space/list-of-llm-consultants/daniel-tannor | $50 to $150 |
|Almog Baku | https://www.linkedin.com/in/almogbaku | https://llmops.space/list-of-llm-consultants/almog-baku | $150 or more |

## List of LLM-Related Events (External) 

- [The AI Conference](https://aiconference.com/) - San Fransico, September 26th-27th 2023
- [World Summit AI](https://worldsummit.ai/) - Amsterdam, October 11th-12th 2023 
- [MLOPS World](https://mlopsworld.com/) - Texas, October 25th-26th 2023

## Beta Program Listings

- [Bench AI](https://bench-ai.com/) - Bench AI is an MLOps platform that automates training and testing for ML models and serves as a management tool for Machine Learning Engineers.
- [Parea AI](https://www.parea.ai/) - Parea AI (YC S23) is the essential developer platform for debugging and monitoring every stage of LLM application development. 
- [LogSpend](https://logspend.com/) - An AI Copilot that Optimizes The Cost and Performance of Your Generative AI Stack


☺️ Our goal is to enable LLMOps companies and potential users/customers to arrive at a win-win situation.

✅ The companies can gain: 
1. Valuable feedback for products or modules that aren't yet ready to be released.
2. Relationships with teams that can consider becoming customers once the product is mature.

✅ The users/customers can gain:
1. Knowledge about tooling & the space
2. Good karma
3. SWAG or in some cases monetary compensation for their time
4. Discounts once the product is ready
5. Ability to influence the product


📝 If you’d like to list your beta program so that members of the LLMOps.Space can join your beta program, please please fill out [👉this👈](https://forms.gle/S37ehh2DP72mJ3SL9) form. 
