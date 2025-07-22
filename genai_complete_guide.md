# Complete Guide: Building Generative AI Applications
## From Beginner to Advanced

---

## Table of Contents
1. [Foundation & Prerequisites](#foundation--prerequisites)
2. [Understanding Generative AI](#understanding-generative-ai)
3. [Development Environment Setup](#development-environment-setup)
4. [Beginner Level: Basic Applications](#beginner-level-basic-applications)
5. [Intermediate Level: Custom Solutions](#intermediate-level-custom-solutions)
6. [Advanced Level: Production Systems](#advanced-level-production-systems)
7. [Specialized Applications](#specialized-applications)
8. [Production Deployment](#production-deployment)
9. [Best Practices & Optimization](#best-practices--optimization)
10. [Resources & Next Steps](#resources--next-steps)

---

## Foundation & Prerequisites

### Essential Skills
- **Python Programming**: Intermediate level (functions, classes, libraries)
- **Machine Learning Basics**: Understanding of neural networks, training, inference
- **API Development**: REST APIs, JSON handling
- **Version Control**: Git and GitHub
- **Cloud Platforms**: Basic AWS/GCP/Azure knowledge

### Mathematical Foundation
- Linear algebra (vectors, matrices)
- Statistics and probability
- Calculus (for optimization understanding)
- Information theory basics

---

## Understanding Generative AI

### Core Concepts

#### 1. Types of Generative Models
- **Large Language Models (LLMs)**: GPT, Claude, LLaMA
- **Diffusion Models**: DALL-E, Stable Diffusion, Midjourney
- **Generative Adversarial Networks (GANs)**: StyleGAN, CycleGAN
- **Variational Autoencoders (VAEs)**: For latent space generation
- **Autoregressive Models**: For sequential data generation

#### 2. Key Terminology
- **Tokens**: Basic units of text processing
- **Context Window**: Maximum input length a model can process
- **Temperature**: Controls randomness in generation
- **Top-k/Top-p Sampling**: Methods for selecting next tokens
- **Fine-tuning**: Adapting pre-trained models for specific tasks
- **Prompt Engineering**: Crafting effective inputs for desired outputs

#### 3. Model Architectures
- **Transformers**: Attention mechanism, encoder-decoder structure
- **GPT Architecture**: Decoder-only, autoregressive generation
- **BERT Architecture**: Encoder-only, bidirectional understanding
- **T5 Architecture**: Text-to-text transfer transformer

---

## Development Environment Setup

### 1. Python Environment
```bash
# Create virtual environment
python -m venv genai_env
source genai_env/bin/activate  # Linux/Mac
genai_env\Scripts\activate     # Windows

# Essential packages
pip install torch transformers datasets
pip install openai anthropic
pip install streamlit gradio
pip install langchain langsmith
pip install huggingface_hub
pip install accelerate bitsandbytes
```

### 2. Development Tools
```bash
# Code quality and development
pip install black isort flake8 mypy
pip install jupyter notebook
pip install python-dotenv
pip install requests fastapi uvicorn
```

### 3. GPU Setup (Optional but Recommended)
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Environment Configuration
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```

---

## Beginner Level: Basic Applications

### Project 1: Simple Chatbot with OpenAI API

#### Basic Implementation
```python
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class SimpleChatbot:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.conversation = []
    
    def chat(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.conversation,
            temperature=0.7,
            max_tokens=150
        )
        
        assistant_reply = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_reply})
        
        return assistant_reply
    
    def reset_conversation(self):
        self.conversation = []

# Usage
bot = SimpleChatbot()
print(bot.chat("Hello! How are you?"))
```

#### Enhanced Version with System Prompts
```python
class AdvancedChatbot(SimpleChatbot):
    def __init__(self, model="gpt-3.5-turbo", system_prompt=None):
        super().__init__(model)
        if system_prompt:
            self.conversation = [{"role": "system", "content": system_prompt}]
    
    def set_personality(self, personality):
        system_prompts = {
            "helpful": "You are a helpful assistant who provides clear and concise answers.",
            "creative": "You are a creative assistant who thinks outside the box.",
            "professional": "You are a professional assistant with expertise in business."
        }
        self.conversation = [{"role": "system", "content": system_prompts.get(personality, system_prompts["helpful"])}]

# Usage
creative_bot = AdvancedChatbot(system_prompt="You are a creative writing assistant.")
```

### Project 2: Text Generation with Hugging Face Transformers

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.generator = pipeline('text-generation', 
                                model=self.model, 
                                tokenizer=self.tokenizer)
    
    def generate(self, prompt, max_length=100, temperature=0.8):
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        result = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return result[0]['generated_text']

# Usage
generator = TextGenerator()
text = generator.generate("The future of artificial intelligence is", max_length=150)
print(text)
```

### Project 3: Simple Streamlit Web Interface

```python
import streamlit as st
from your_chatbot_module import AdvancedChatbot

st.title("🤖 My First GenAI Chatbot")

# Initialize chatbot in session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = AdvancedChatbot()

# Sidebar for settings
st.sidebar.title("Settings")
personality = st.sidebar.selectbox(
    "Choose personality:",
    ["helpful", "creative", "professional"]
)

if st.sidebar.button("Reset Conversation"):
    st.session_state.chatbot.reset_conversation()
    st.session_state.chatbot.set_personality(personality)

# Chat interface
user_input = st.text_input("You:", key="user_input")

if user_input:
    with st.spinner("Thinking..."):
        response = st.session_state.chatbot.chat(user_input)
    
    st.text_area("Bot:", value=response, height=200)

# Display conversation history
if st.checkbox("Show conversation history"):
    st.write(st.session_state.chatbot.conversation)
```

---

## Intermediate Level: Custom Solutions

### Project 4: RAG (Retrieval-Augmented Generation) System

#### Document Processing and Embedding
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict

class DocumentStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def add_documents(self, documents: List[str]):
        """Add documents to the store and create embeddings"""
        self.documents.extend(documents)
        
        # Create embeddings
        doc_embeddings = self.embedding_model.encode(documents)
        
        if self.embeddings is None:
            self.embeddings = doc_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embeddings])
        
        # Create/update FAISS index
        dimension = doc_embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(doc_embeddings)
        self.index.add(doc_embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def save(self, filepath: str):
        """Save the document store"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        
        if self.index:
            faiss.write_index(self.index, filepath + '.index')
    
    def load(self, filepath: str):
        """Load the document store"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
        
        try:
            self.index = faiss.read_index(filepath + '.index')
        except:
            print("Index file not found, creating new index...")
            if self.embeddings is not None:
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings)
```

#### RAG Pipeline Implementation
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

class RAGSystem:
    def __init__(self, document_store: DocumentStore, llm_model="gpt-3.5-turbo"):
        self.document_store = document_store
        self.llm_model = llm_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_documents_from_files(self, file_paths: List[str]):
        """Load and process documents from files"""
        all_chunks = []
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            all_chunks.extend(chunks)
        
        self.document_store.add_documents(all_chunks)
        print(f"Loaded {len(all_chunks)} document chunks")
    
    def query(self, question: str, k: int = 3) -> str:
        """Query the RAG system"""
        # Retrieve relevant documents
        relevant_docs = self.document_store.search(question, k=k)
        
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context
        context = "\n\n".join([doc['document'] for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer: """
        
        # Generate response
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Usage example
doc_store = DocumentStore()
rag_system = RAGSystem(doc_store)

# Load your documents
rag_system.load_documents_from_files(['document1.txt', 'document2.txt'])

# Query the system
answer = rag_system.query("What is the main topic discussed in the documents?")
print(answer)
```

### Project 5: Fine-tuning a Small Language Model

```python
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

class FineTuner:
    def __init__(self, base_model="microsoft/DialoGPT-small"):
        self.model_name = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, conversations: List[List[str]]):
        """Prepare dataset for training"""
        # Format conversations
        formatted_texts = []
        for conversation in conversations:
            formatted_conv = ""
            for i, turn in enumerate(conversation):
                if i % 2 == 0:  # User turn
                    formatted_conv += f"User: {turn}\n"
                else:  # Assistant turn
                    formatted_conv += f"Assistant: {turn}\n"
            formatted_conv += self.tokenizer.eos_token
            formatted_texts.append(formatted_conv)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, dataset, output_dir="./fine-tuned-model", epochs=3):
        """Fine-tune the model"""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're not doing masked language modeling
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        trainer.train()
        trainer.save_model()
    
    def generate_response(self, prompt: str, max_length=100):
        """Generate response with fine-tuned model"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage example
conversations = [
    ["Hello!", "Hi there! How can I help you today?"],
    ["What's the weather like?", "I don't have access to weather data, but you can check a weather app!"],
    # Add more conversation examples...
]

fine_tuner = FineTuner()
dataset = fine_tuner.prepare_dataset(conversations)
fine_tuner.train(dataset)
```

### Project 6: Multi-Modal AI Application

```python
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

class MultiModalAI:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.text_generator = pipeline('text-generation', model='gpt2')
    
    def image_to_text(self, image_path_or_url: str, prompt: str = None) -> str:
        """Generate text description from image"""
        # Load image
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path_or_url).convert('RGB')
        
        # Process image
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")
        
        # Generate caption
        out = self.model.generate(**inputs, max_length=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    def create_story_from_image(self, image_path_or_url: str) -> str:
        """Create a story based on image description"""
        # Get image description
        description = self.image_to_text(image_path_or_url)
        
        # Generate story
        story_prompt = f"Based on this scene: '{description}', write a short story:"
        story = self.text_generator(
            story_prompt,
            max_length=200,
            temperature=0.8,
            do_sample=True
        )
        
        return story[0]['generated_text']

# Usage
multimodal_ai = MultiModalAI()
description = multimodal_ai.image_to_text("path/to/your/image.jpg")
story = multimodal_ai.create_story_from_image("path/to/your/image.jpg")
```

---

## Advanced Level: Production Systems

### Project 7: Scalable FastAPI Backend

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import redis
import json
from datetime import datetime, timedelta
import uuid

app = FastAPI(title="GenAI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching and rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime

class GenerationRequest(BaseModel):
    prompt: str
    model_type: str = "text"
    parameters: Optional[dict] = {}

# Rate limiting
async def rate_limit(request_id: str):
    """Simple rate limiting"""
    key = f"rate_limit:{request_id}"
    current = redis_client.get(key)
    
    if current is None:
        redis_client.setex(key, 60, 1)  # 1 request per minute
        return True
    
    if int(current) >= 10:  # Max 10 requests per minute
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    redis_client.incr(key)
    return True

# Session management
class SessionManager:
    @staticmethod
    def get_or_create_session(session_id: Optional[str]) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        return session_id
    
    @staticmethod
    def get_conversation_history(session_id: str) -> List[dict]:
        history = redis_client.get(f"session:{session_id}")
        if history:
            return json.loads(history)
        return []
    
    @staticmethod
    def save_conversation(session_id: str, conversation: List[dict]):
        redis_client.setex(
            f"session:{session_id}",
            3600,  # 1 hour expiry
            json.dumps(conversation)
        )

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(lambda: rate_limit("default"))
):
    """Chat with the AI"""
    try:
        session_id = SessionManager.get_or_create_session(request.session_id)
        
        # Get conversation history
        conversation = SessionManager.get_conversation_history(session_id)
        conversation.append({"role": "user", "content": request.message})
        
        # Generate response (replace with your AI model)
        ai_response = await generate_ai_response(
            conversation,
            request.temperature,
            request.max_tokens
        )
        
        conversation.append({"role": "assistant", "content": ai_response})
        
        # Save conversation in background
        background_tasks.add_task(
            SessionManager.save_conversation,
            session_id,
            conversation
        )
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_content(request: GenerationRequest):
    """Generate content based on prompt and type"""
    try:
        if request.model_type == "text":
            result = await generate_text(request.prompt, request.parameters)
        elif request.model_type == "code":
            result = await generate_code(request.prompt, request.parameters)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        return {"result": result, "type": request.model_type}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    history = SessionManager.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history}

# AI Generation functions (implement with your models)
async def generate_ai_response(conversation: List[dict], temperature: float, max_tokens: int) -> str:
    """Generate AI response - implement with your model"""
    # This is a placeholder - integrate with your actual model
    await asyncio.sleep(0.1)  # Simulate processing time
    return "This is a placeholder response. Integrate with your actual AI model."

async def generate_text(prompt: str, parameters: dict) -> str:
    """Generate text content"""
    # Implement with your text generation model
    return f"Generated text for: {prompt}"

async def generate_code(prompt: str, parameters: dict) -> str:
    """Generate code content"""
    # Implement with your code generation model
    return f"# Generated code for: {prompt}\nprint('Hello, World!')"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Project 8: Advanced Prompt Engineering Framework

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from enum import Enum

class PromptType(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    ROLE_BASED = "role_based"

@dataclass
class Example:
    input: str
    output: str
    explanation: Optional[str] = None

@dataclass
class PromptTemplate:
    template: str
    variables: List[str]
    type: PromptType
    examples: Optional[List[Example]] = None
    system_message: Optional[str] = None

class PromptBuilder:
    """Advanced prompt building with various techniques"""
    
    def __init__(self):
        self.templates = {}
    
    def register_template(self, name: str, template: PromptTemplate):
        """Register a prompt template"""
        self.templates[name] = template
    
    def build_zero_shot_prompt(self, instruction: str, input_data: str) -> str:
        """Build a zero-shot prompt"""
        return f"""Instruction: {instruction}

Input: {input_data}

Output:"""
    
    def build_few_shot_prompt(self, instruction: str, examples: List[Example], input_data: str) -> str:
        """Build a few-shot prompt with examples"""
        prompt = f"Instruction: {instruction}\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example.input}\n"
            prompt += f"Output: {example.output}\n"
            if example.explanation:
                prompt += f"Explanation: {example.explanation}\n"
            prompt += "\n"
        
        prompt += f"Now solve this:\nInput: {input_data}\nOutput:"
        return prompt
    
    def build_chain_of_thought_prompt(self, instruction: str, input_data: str, examples: Optional[List[Example]] = None) -> str:
        """Build a chain-of-thought prompt"""
        base_instruction = f"{instruction}\n\nLet's think step by step:"
        
        if examples:
            return self.build_few_shot_prompt(base_instruction, examples, input_data)
        else:
            return self.build_zero_shot_prompt(base_instruction, input_data)
    
    def build_role_based_prompt(self, role: str, instruction: str, input_data: str, context: Optional[str] = None) -> str:
        """Build a role-based prompt"""
        prompt = f"You are {role}.\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += f"Task: {instruction}\n\n"
        prompt += f"Input: {input_data}\n\n"
        prompt += "Please provide your response:"
        
        return prompt
    
    def build_from_template(self, template_name: str, **kwargs) -> str:
        """Build prompt from registered template"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
        
        # Check if all required variables are provided
        missing_vars = set(template.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Build prompt based on type
        if template.type == PromptType.ZERO_SHOT:
            return self.build_zero_shot_prompt(kwargs['instruction'], kwargs['input'])
        elif template.type == PromptType.FEW_SHOT:
            return self.build_few_shot_prompt(kwargs['instruction'], template.examples, kwargs['input'])
        elif template.type == PromptType.CHAIN_OF_THOUGHT:
            return self.build_chain_of_thought_prompt(kwargs['instruction'], kwargs['input'], template.examples)
        elif template.type == PromptType.ROLE_BASED:
            return self.build_role_based_prompt(kwargs['role'], kwargs['instruction'], kwargs['input'], kwargs.get('context'))
        
        # Fallback to template string formatting
        return template.template.format(**kwargs)

class PromptOptimizer:
    """Optimize prompts for better performance"""
    
    def __init__(self, ai_model):
        self.model = ai_model
        self.performance_history = []
    
    def test_prompt_variations(self, base_prompt: str, variations: List[str], test_cases: List[Dict]) -> Dict:
        """Test different prompt variations and return performance metrics"""
        results = {}
        
        for i, variation in enumerate([base_prompt] + variations):
            results[f"variation_{i}"] = {
                'prompt': variation,
                'scores': [],
                'responses': []
            }
            
            for test_case in test_cases:
                # Generate response
                response = self.model.generate(variation.format(**test_case['input']))
                
                # Calculate score (implement your scoring logic)
                score = self._calculate_score(response, test_case['expected'])
                
                results[f"variation_{i}"]["scores"].append(score)
                results[f"variation_{i}"]["responses"].append(response)
            
            # Calculate average score
            results[f"variation_{i}"]["avg_score"] = sum(results[f"variation_{i}"]["scores"]) / len(results[f"variation_{i}"]["scores"])
        
        return results
    
    def _calculate_score(self, response: str, expected: str) -> float:
        """Calculate similarity score between response and expected output"""
        # Implement your scoring logic here (e.g., BLEU, ROUGE, semantic similarity)
        # This is a placeholder
        return 0.8

# Usage Example
def setup_prompt_templates():
    builder = PromptBuilder()
    
    # Register a code generation template
    code_template = PromptTemplate(
        template="""You are an expert {language} programmer.

Task: {task}

Requirements:
{requirements}

Please provide clean, well-commented code:""",
        variables=["language", "task", "requirements"],
        type=PromptType.ROLE_BASED,
        system_message="You are a helpful coding assistant."
    )
    
    builder.register_template("code_generation", code_template)
    
    # Register a few-shot classification template
    classification_examples = [
        Example("The movie was fantastic!", "positive", "The word 'fantastic' indicates positive sentiment"),
        Example("I hate this product.", "negative", "The word 'hate' indicates negative sentiment"),
        Example("The weather is okay.", "neutral", "The word 'okay' indicates neutral sentiment")
    ]
    
    sentiment_template = PromptTemplate(
        template="",
        variables=["instruction", "input"],
        type=PromptType.FEW_SHOT,
        examples=classification_examples
    )
    
    builder.register_template("sentiment_analysis", sentiment_template)
    
    return builder

# Usage
builder = setup_prompt_templates()

# Generate code prompt
code_prompt = builder.build_from_template(
    "code_generation",
    language="Python",
    task="Create a function to calculate fibonacci numbers",
    requirements="- Use recursion\n- Include error handling\n- Add docstring"
)

print(code_prompt)
```

---

## Specialized Applications

### Project 9: Code Generation Assistant

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast
import subprocess
import tempfile
import os

class CodeGenerator:
    def __init__(self, model_name="microsoft/CodeGPT-small-py"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_code(self, prompt: str, max_length: int = 200, language: str = "python") -> str:
        """Generate code based on prompt"""
        formatted_prompt = f"# {prompt}\n"
        
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code
    
    def validate_python_syntax(self, code: str) -> tuple:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return True, "Syntax is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def execute_code(self, code: str, timeout: int = 10) -> tuple:
        """Safely execute Python code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Code execution timed out"
        except Exception as e:
            return False, f"Execution error: {e}"
    
    def improve_code(self, code: str, feedback: str) -> str:
        """Improve code based on feedback"""
        prompt = f"""
Improve the following code based on the feedback:

Original code:
```python
{code}
```

Feedback: {feedback}

Improved code:
```python
"""
        return self.generate_code(prompt)

class CodeReviewer:
    """Automated code review assistant"""
    
    def __init__(self):
        self.common_issues = {
            'naming': self._check_naming_conventions,
            'complexity': self._check_complexity,
            'documentation': self._check_documentation,
            'security': self._check_security_issues
        }
    
    def review_code(self, code: str) -> Dict[str, List[str]]:
        """Perform comprehensive code review"""
        issues = {}
        
        for check_name, check_function in self.common_issues.items():
            found_issues = check_function(code)
            if found_issues:
                issues[check_name] = found_issues
        
        return issues
    
    def _check_naming_conventions(self, code: str) -> List[str]:
        """Check naming conventions"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for camelCase variables (should be snake_case in Python)
            if 'def ' in line and any(c.isupper() for c in line.split('def ')[1].split('(')[0]):
                issues.append(f"Line {i}: Function name should use snake_case")
        
        return issues
    
    def _check_complexity(self, code: str) -> List[str]:
        """Check code complexity"""
        issues = []
        lines = code.split('\n')
        
        # Simple check for deeply nested code
        for i, line in enumerate(lines, 1):
            indent_level = (len(line) - len(line.lstrip())) // 4
            if indent_level > 3:
                issues.append(f"Line {i}: Code is deeply nested (level {indent_level})")
        
        return issues
    
    def _check_documentation(self, code: str) -> List[str]:
        """Check documentation"""
        issues = []
        
        if 'def ' in code and '"""' not in code and "'''" not in code:
            issues.append("Functions should have docstrings")
        
        return issues
    
    def _check_security_issues(self, code: str) -> List[str]:
        """Check for common security issues"""
        issues = []
        dangerous_functions = ['eval', 'exec', 'input', '__import__']
        
        for func in dangerous_functions:
            if func in code:
                issues.append(f"Potentially dangerous function '{func}' detected")
        
        return issues

# Usage example
code_gen = CodeGenerator()
reviewer = CodeReviewer()

# Generate code
prompt = "Create a function to sort a list of dictionaries by a given key"
generated_code = code_gen.generate_code(prompt)

# Review the code
review_results = reviewer.review_code(generated_code)

# Validate syntax
is_valid, message = code_gen.validate_python_syntax(generated_code)

print(f"Generated Code:\n{generated_code}")
print(f"\nSyntax Valid: {is_valid}")
print(f"Message: {message}")
print(f"\nReview Results: {review_results}")
```

### Project 10: Content Generation Pipeline

```python
from typing import List, Dict, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ContentRequest:
    content_type: str
    topic: str
    target_audience: str
    tone: str
    length: str
    keywords: List[str]
    additional_requirements: Optional[str] = None

@dataclass
class GeneratedContent:
    content: str
    title: str
    meta_description: str
    keywords: List[str]
    word_count: int
    readability_score: float
    timestamp: datetime

class ContentGenerator:
    def __init__(self):
        self.content_templates = {
            'blog_post': self._generate_blog_post,
            'product_description': self._generate_product_description,
            'social_media': self._generate_social_media,
            'email_newsletter': self._generate_email_newsletter,
            'ad_copy': self._generate_ad_copy
        }
    
    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """Generate content based on request"""
        if request.content_type not in self.content_templates:
            raise ValueError(f"Unsupported content type: {request.content_type}")
        
        generator_func = self.content_templates[request.content_type]
        content_data = await generator_func(request)
        
        # Calculate readability score
        readability_score = self._calculate_readability(content_data['content'])
        
        return GeneratedContent(
            content=content_data['content'],
            title=content_data['title'],
            meta_description=content_data['meta_description'],
            keywords=request.keywords,
            word_count=len(content_data['content'].split()),
            readability_score=readability_score,
            timestamp=datetime.now()
        )
    
    async def _generate_blog_post(self, request: ContentRequest) -> Dict:
        """Generate blog post content"""
        prompt = f"""
Write a {request.length} blog post about {request.topic} for {request.target_audience}.

Tone: {request.tone}
Keywords to include: {', '.join(request.keywords)}

Requirements:
- Engaging title
- Clear introduction
- Well-structured body with subheadings
- Conclusion with call-to-action
- SEO-friendly meta description

{request.additional_requirements or ''}
"""
        
        # Generate content using your AI model
        content = await self._call_ai_model(prompt)
        
        # Extract title and meta description
        title = self._extract_title(content)
        meta_description = self._generate_meta_description(content, request.keywords)
        
        return {
            'content': content,
            'title': title,
            'meta_description': meta_description
        }
    
    async def _generate_product_description(self, request: ContentRequest) -> Dict:
        """Generate product description"""
        prompt = f"""
Write a compelling product description for {request.topic}.

Target audience: {request.target_audience}
Tone: {request.tone}
Length: {request.length}
Keywords: {', '.join(request.keywords)}

Focus on:
- Key features and benefits
- Unique selling points
- Emotional appeal
- Call-to-action

{request.additional_requirements or ''}
"""
        
        content = await self._call_ai_model(prompt)
        title = f"{request.topic} - Product Description"
        meta_description = self._generate_meta_description(content, request.keywords)
        
        return {
            'content': content,
            'title': title,
            'meta_description': meta_description
        }
    
    async def _generate_social_media(self, request: ContentRequest) -> Dict:
        """Generate social media content"""
        platform = request.additional_requirements or "general social media"
        
        prompt = f"""
Create engaging social media content about {request.topic} for {platform}.

Target audience: {request.target_audience}
Tone: {request.tone}
Length: {request.length}

Include:
- Attention-grabbing hook
- Relevant hashtags
- Call-to-action
- Keywords: {', '.join(request.keywords)}
"""
        
        content = await self._call_ai_model(prompt)
        title = f"Social Media Post: {request.topic}"
        meta_description = content[:150] + "..."
        
        return {
            'content': content,
            'title': title,
            'meta_description': meta_description
        }
    
    async def _call_ai_model(self, prompt: str) -> str:
        """Call AI model to generate content"""
        # Implement with your preferred AI model (OpenAI, Anthropic, etc.)
        # This is a placeholder
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Generated content for: {prompt[:50]}..."
    
    def _extract_title(self, content: str) -> str:
        """Extract or generate title from content"""
        lines = content.split('\n')
        for line in lines:
            if line.strip() and len(line.strip()) < 100:
                return line.strip()
        return "Generated Content"
    
    def _generate_meta_description(self, content: str, keywords: List[str]) -> str:
        """Generate SEO-friendly meta description"""
        # Simple implementation - take first 155 characters and ensure keywords are included
        sentences = content.split('. ')
        meta = sentences[0]
        
        # Try to include keywords
        for keyword in keywords[:3]:  # Include up to 3 keywords
            if keyword.lower() not in meta.lower() and len(meta) < 120:
                meta += f" {keyword}"
        
        return meta[:155] + ("..." if len(meta) > 155 else "")
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score))
    
    def _count_syllables(self, text: str) -> int:
        """Simple syllable counting"""
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in text.lower():
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        return max(1, syllables)

class ContentOptimizer:
    """Optimize content for SEO and engagement"""
    
    def __init__(self):
        self.optimization_rules = {
            'seo': self._optimize_seo,
            'readability': self._optimize_readability,
            'engagement': self._optimize_engagement
        }
    
    def optimize_content(self, content: GeneratedContent, optimization_type: str) -> GeneratedContent:
        """Optimize content based on type"""
        if optimization_type not in self.optimization_rules:
            raise ValueError(f"Unsupported optimization type: {optimization_type}")
        
        optimizer_func = self.optimization_rules[optimization_type]
        optimized_content = optimizer_func(content)
        
        return optimized_content
    
    def _optimize_seo(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for SEO"""
        # Add keyword density optimization
        # Improve meta description
        # Enhance title tags
        # This is a simplified implementation
        
        optimized_content = content.content
        
        # Ensure keywords appear in strategic locations
        for keyword in content.keywords:
            if keyword.lower() not in content.title.lower():
                # Try to incorporate keyword into title
                pass
        
        return content
    
    def _optimize_readability(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for readability"""
        # Simplify complex sentences
        # Add transition words
        # Improve paragraph structure
        return content
    
    def _optimize_engagement(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for engagement"""
        # Add compelling hooks
        # Include call-to-actions
        # Enhance emotional appeal
        return content

# Usage example
async def main():
    generator = ContentGenerator()
    optimizer = ContentOptimizer()
    
    # Create content request
    request = ContentRequest(
        content_type="blog_post",
        topic="Sustainable Living Tips",
        target_audience="environmentally conscious millennials",
        tone="friendly and informative",
        length="800-1000 words",
        keywords=["sustainable living", "eco-friendly", "green lifestyle", "environmental impact"],
        additional_requirements="Include practical tips and actionable advice"
    )
    
    # Generate content
    content = await generator.generate_content(request)
    
    # Optimize content
    optimized_content = optimizer.optimize_content(content, "seo")
    
    print(f"Title: {content.title}")
    print(f"Word Count: {content.word_count}")
    print(f"Readability Score: {content.readability_score}")
    print(f"Meta Description: {content.meta_description}")
    print(f"\nContent:\n{content.content}")

# Run the example
# asyncio.run(main())
```

---

## Production Deployment

### Docker Configuration

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  genai-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/genai_db
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=genai_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - genai-app
    restart: unless-stopped

volumes:
  postgres_data:
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-app
  labels:
    app: genai-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genai-app
  template:
    metadata:
      labels:
        app: genai-app
    spec:
      containers:
      - name: genai-app
        image: your-registry/genai-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: genai-service
spec:
  selector:
    app: genai-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Monitoring and Logging

**monitoring.py:**
```python
import time
import logging
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest
import psutil
import asyncio

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def monitor_request(self, func):
        """Decorator to monitor API requests"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(method='POST', endpoint=func.__name__).inc()
                return result
                
            except Exception as e:
                ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                self.logger.error(f"Error in {func.__name__}: {e}")
                raise
                
            finally:
                REQUEST_LATENCY.observe(time.time() - start_time)
                
        return wrapper
    
    def monitor_model_inference(self, func):
        """Decorator to monitor model inference"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
                
            finally:
                MODEL_INFERENCE_TIME.observe(time.time() - start_time)
                
        return wrapper
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }

class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('genai_app.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_request(self, request_data, response_data, duration):
        """Log request details"""
        self.logger.info(f"Request processed in {duration:.2f}s")
        self.logger.debug(f"Request: {request_data}")
        self.logger.debug(f"Response: {response_data}")
    
    def log_error(self, error, context):
        """Log error with context"""
        self.logger.error(f"Error: {error}, Context: {context}")

# Usage in your FastAPI app
monitor = PerformanceMonitor()
app_logger = Logger(__name__)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health_check():
    """Enhanced health check with system metrics"""
    system_metrics = monitor.get_system_metrics()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_metrics": system_metrics
    }
```

---

## Best Practices & Optimization

### Performance Optimization

```python
import asyncio
import aioredis
from contextlib import asynccontextmanager
import torch
from transformers import pipeline
import gc

class ModelCache:
    """Efficient model caching and management"""
    
    def __init__(self):
        self.models = {}
        self.max_models = 3  # Maximum models in memory
        self.usage_count = {}
    
    async def get_model(self, model_name: str, model_type: str):
        """Get model from cache or load if not cached"""
        cache_key = f"{model_name}_{model_type}"
        
        if cache_key in self.models:
            self.usage_count[cache_key] += 1
            return self.models[cache_key]
        
        # Load model
        if len(self.models) >= self.max_models:
            self._evict_least_used_model()
        
        model = await self._load_model(model_name, model_type)
        self.models[cache_key] = model
        self.usage_count[cache_key] = 1
        
        return model
    
    def _evict_least_used_model(self):
        """Remove least used model from cache"""
        if not self.models:
            return
        
        least_used_key = min(self.usage_count.keys(), key=lambda k: self.usage_count[k])
        del self.models[least_used_key]
        del self.usage_count[least_used_key]
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def _load_model(self, model_name: str, model_type: str):
        """Load model based on type"""
        if model_type == "text-generation":
            return pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        elif model_type == "text-classification":
            return pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class RequestBatcher:
    """Batch requests for efficient processing"""
    
    def __init__(self, batch_size: int = 4, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.batch_event = asyncio.Event()
    
    async def add_request(self, request_data):
        """Add request to batch"""
        future = asyncio.Future()
        self.pending_requests.append((request_data, future))
        
        if len(self.pending_requests) >= self.batch_size:
            self.batch_event.set()
        
        return await future
    
    async def process_batches(self, model_func):
        """Process batched requests"""
        while True:
            # Wait for batch to fill or timeout
            try:
                await asyncio.wait_for(self.batch_event.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                pass
            
            if self.pending_requests:
                # Process current batch
                batch = self.pending_requests[:self.batch_size]
                self.pending_requests = self.pending_requests[self.batch_size:]
                
                # Extract requests and futures
                requests, futures = zip(*batch)
                
                try:
                    # Process batch
                    results = await model_func(list(requests))
                    
                    # Return results to futures
                    for future, result in zip(futures, results):
                        future.set_result(result)
                        
                except Exception as e:
                    # Set exception for all futures
                    for future in futures:
                        future.set_exception(e)
                
                self.batch_event.clear()

class ResponseCache:
    """Redis-based response caching"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
        self.default_ttl = 3600  # 1 hour
    
    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def get(self, key: str):
        """Get cached response"""
        if not self.redis:
            await self.connect()
        
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    async def set(self, key: str, value: dict, ttl: int = None):
        """Cache response"""
        if not self.redis:
            await self.connect()
        
        await self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(value)
        )
    
    def generate_cache_key(self, request_data: dict) -> str:
        """Generate cache key from request"""
        # Create deterministic key from request data
        key_data = {k: v for k, v in sorted(request_data.items())}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
```

### Security Best Practices

```python
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib

class SecurityManager:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current user from token"""
        payload = self.verify_token(credentials.credentials)
        return payload.get("sub")
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '/', '\\']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    def rate_limit_key(self, identifier: str, window: str = "hour") -> str:
        """Generate rate limit key"""
        timestamp = datetime.utcnow()
        
        if window == "minute":
            time_window = timestamp.strftime("%Y%m%d%H%M")
        elif window == "hour":
            time_window = timestamp.strftime("%Y%m%d%H")
        elif window == "day":
            time_window = timestamp.strftime("%Y%m%d")
        
        return f"rate_limit:{identifier}:{time_window}"

# Input validation
from pydantic import BaseModel, validator
import re

class SecurePromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.7
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v) > 2000:  # Limit prompt length
            raise ValueError('Prompt too long')
        
        # Check for potential prompt injection
        injection_patterns = [
            r'ignore.*previous.*instructions',
            r'forget.*instructions',
            r'new.*instructions',
            r'system.*prompt'
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, v.lower()):
                raise ValueError('Potentially unsafe prompt detected')
        
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if not 1 <= v <= 2000:
            raise ValueError('Max tokens must be between 1 and 2000')
        return v

# Usage in FastAPI
security_manager = SecurityManager()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for all requests"""
    
    # Add security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response
```

---

## Resources & Next Steps

### Essential Libraries and Frameworks

**Core AI/ML Libraries:**
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for deep learning
- `tensorflow` - Google's ML framework
- `langchain` - Framework for LLM applications
- `llamaindex` - Data framework for LLM applications
- `sentence-transformers` - Sentence embeddings
- `openai` - OpenAI API client
- `anthropic` - Anthropic Claude API client

**Web Development:**
- `fastapi` - Modern web framework
- `streamlit` - Rapid prototyping for ML apps
- `gradio` - User-friendly ML interfaces
- `flask` - Lightweight web framework
- `django` - Full-featured web framework

**Data and Storage:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `faiss` - Similarity search
- `pinecone` - Vector database
- `redis` - In-memory database
- `postgresql` - Relational database

**Deployment and Monitoring:**
- `docker` - Containerization
- `kubernetes` - Container orchestration
- `prometheus` - Monitoring
- `grafana` - Visualization
- `nginx` - Web server and load balancer

### Learning Path

**Phase 1: Foundation (Weeks 1-4)**
1. Complete Python fundamentals
2. Understand basic ML concepts
3. Explore transformer architecture
4. Build simple chatbot projects
5. Learn prompt engineering basics

**Phase 2: Intermediate (Weeks 5-12)**
1. Implement RAG systems
2. Fine-tune small models
3. Build multi-modal applications
4. Create web interfaces
5. Learn vector databases

**Phase 3: Advanced (Weeks 13-24)**
1. Production deployment
2. Performance optimization
3. Security implementation
4. Monitoring and logging
5. Scaling strategies

**Phase 4: Specialization (Ongoing)**
1. Domain-specific applications
2. Custom model architectures
3. Research and development
4. Contributing to open source
5. Building commercial products

### Key Resources

**Documentation:**
- [Hugging Face Documentation](https://huggingface.co/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://docs.langchain.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

**Courses and Tutorials:**
- CS229 Machine Learning (Stanford)
- Deep Learning Specialization (Coursera)
- Practical Deep Learning for Coders (fast.ai)
- Transformers Course (Hugging Face)

**Communities:**
- Hugging Face Community
- r/MachineLearning
- AI/ML Discord servers
- GitHub repositories

### Common Pitfalls to Avoid

1. **Not understanding the business problem first**
2. **Over-engineering solutions**
3. **Ignoring data quality and preprocessing**
4. **Insufficient testing and validation**
5. **Poor error handling and monitoring**
6. **Neglecting security considerations**
7. **Not planning for scalability**
8. **Inadequate documentation**

### Next Steps

1. **Start with a simple project** - Build a basic chatbot or text generator
2. **Iterate and improve** - Add features incrementally
3. **Focus on one domain** - Become expert in a specific application area
4. **Join communities** - Learn from others and share your work
5. **Keep learning** - AI/ML field evolves rapidly
6. **Build a portfolio** - Showcase your projects on GitHub
7. **Consider commercial applications** - Look for business opportunities

Remember: The key to success in Generative AI is consistent practice, continuous learning, and building real projects. Start simple, be patient with yourself, and gradually tackle more complex challenges as you build confidence and expertise.

---

*This guide provides a comprehensive roadmap for building Generative AI applications. Each section builds upon previous knowledge, ensuring a solid foundation for creating production-ready systems. Remember to practice regularly and adapt the examples to your specific use cases.*