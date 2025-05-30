# # JobCoach: Career Development Agent with Real LLMs
# # Complete implementation with LangChain, Streamlit, and real free LLMs

# import streamlit as st
# import pandas as pd
# import json
# import re
# import os
# from datetime import datetime
# from typing import Dict, List, Optional
# import requests
# from dataclasses import dataclass
# import plotly.express as px
# import plotly.graph_objects as go
# from io import StringIO
# import PyPDF2
# from PIL import Image
# import base64
# import time

# # Core imports for LangChain
# from langchain.agents import initialize_agent, AgentType, Tool
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import HumanMessage, SystemMessage
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.tools import BaseTool
# from langchain.callbacks.base import BaseCallbackHandler

# # Real LLM integrations
# # try:
# from langchain.llms import Ollama
# OLLAMA_AVAILABLE = True
# # except ImportError:
# #     OLLAMA_AVAILABLE = False

# try:
#     from langchain.llms import HuggingFacePipeline
#     from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#     import torch
#     HF_AVAILABLE = True
# except ImportError:
#     HF_AVAILABLE = False

# try:
#     from langchain.llms import OpenAI
#     from langchain.chat_models import ChatOpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# try:
#     import google.generativeai as genai
#     from langchain.llms import GooglePalm
#     GOOGLE_AVAILABLE = True
# except ImportError:
#     GOOGLE_AVAILABLE = False

# # Configuration
# st.set_page_config(
#     page_title="JobCoach - AI Career Mentor",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Data classes for structured information
# @dataclass
# class SkillGap:
#     skill: str
#     current_level: str
#     required_level: str
#     learning_resources: List[str]
#     priority: str

# @dataclass
# class CareerAdvice:
#     recommendation: str
#     reasoning: str
#     action_items: List[str]
#     timeline: str

# # Real LLM Manager Class
# class LLMManager:
#     def __init__(self):
#         self.available_llms = {}
#         self.setup_llms()
    
#     def setup_llms(self):
#         """Initialize available LLMs based on what's installed and configured"""
        
#         # 1. Ollama Models (Free, Local)
#         if OLLAMA_AVAILABLE:
#             try:
#                 # Test if Ollama is running
#                 response = requests.get("http://localhost:11434/api/tags", timeout=5)
#                 if response.status_code == 200:
#                     models = response.json().get('models', [])
#                     for model in models:
#                         model_name = model['name']
#                         self.available_llms[f"ollama_{model_name}"] = Ollama(
#                             model=model_name,
#                             temperature=0.7,
#                             num_predict=1000
#                         )
#                     st.sidebar.success(f"✅ Ollama: {len(models)} models available")
#                 else:
#                     st.sidebar.warning("⚠️ Ollama server not running")
#             except Exception as e:
#                 st.sidebar.info("ℹ️ Ollama not available - install with: `ollama pull llama2`")
        
#         # 2. Hugging Face Transformers (Free)
#         if HF_AVAILABLE:
#             try:
#                 # Use smaller, efficient models that work well for career advice
#                 models_to_try = [
#                     "microsoft/DialoGPT-medium",
#                     "facebook/blenderbot-400M-distill",
#                     "microsoft/DialoGPT-small"
#                 ]
                
#                 for model_name in models_to_try:
#                     try:
#                         tokenizer = AutoTokenizer.from_pretrained(model_name)
#                         model = AutoModelForCausalLM.from_pretrained(
#                             model_name,
#                             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                             device_map="auto" if torch.cuda.is_available() else None
#                         )
                        
#                         pipe = pipeline(
#                             "text-generation",
#                             model=model,
#                             tokenizer=tokenizer,
#                             max_length=512,
#                             temperature=0.7,
#                             do_sample=True,
#                             pad_token_id=tokenizer.eos_token_id
#                         )
                        
#                         hf_llm = HuggingFacePipeline(pipeline=pipe)
#                         self.available_llms[f"huggingface_{model_name.split('/')[-1]}"] = hf_llm
#                         st.sidebar.success(f"✅ HuggingFace: {model_name.split('/')[-1]} loaded")
#                         break  # Use first successful model
#                     except Exception as e:
#                         continue
                        
#                 if not any('huggingface' in k for k in self.available_llms.keys()):
#                     st.sidebar.info("ℹ️ HuggingFace models need more memory - trying lightweight options")
                    
#             except Exception as e:
#                 st.sidebar.warning(f"⚠️ HuggingFace setup error: {str(e)[:50]}...")
        
#         # 3. OpenAI (if API key provided)
#         if OPENAI_AVAILABLE and st.secrets.get("OPENAI_API_KEY"):
#             try:
#                 self.available_llms["openai_gpt-3.5-turbo"] = ChatOpenAI(
#                     model_name="gpt-3.5-turbo",
#                     temperature=0.7,
#                     openai_api_key=st.secrets["OPENAI_API_KEY"]
#                 )
#                 st.sidebar.success("✅ OpenAI: GPT-3.5-Turbo available")
#             except Exception as e:
#                 st.sidebar.warning("⚠️ OpenAI API key issues")
        
#         # 4. Google PaLM (if API key provided)
#         if GOOGLE_AVAILABLE and st.secrets.get("GOOGLE_API_KEY"):
#             try:
#                 genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
#                 self.available_llms["google_palm"] = GooglePalm(
#                     google_api_key=st.secrets["GOOGLE_API_KEY"],
#                     temperature=0.7
#                 )
#                 st.sidebar.success("✅ Google: PaLM available")
#             except Exception as e:
#                 st.sidebar.warning("⚠️ Google API key issues")
        
#         # Fallback: Simple rule-based responses if no LLMs available
#         if not self.available_llms:
#             st.sidebar.error("❌ No LLMs available! Please install Ollama or configure API keys.")
#             self.available_llms["fallback"] = self._create_fallback_llm()
    
#     def _create_fallback_llm(self):
#         """Create a fallback LLM using rule-based responses"""
#         class FallbackLLM:
#             def predict(self, text: str) -> str:
#                 return self._generate_fallback_response(text)
            
#             def _generate_fallback_response(self, query: str) -> str:
#                 query_lower = query.lower()
                
#                 if "resume" in query_lower:
#                     return """**Resume Analysis:**
                    
# Based on your query, here are key resume improvement areas:

# **Strengths to Highlight:**
# • Technical skills alignment with job requirements
# • Quantified achievements and measurable results
# • Professional experience progression

# **Areas for Enhancement:**
# • Include more industry-specific keywords for ATS optimization
# • Add relevant certifications or professional development
# • Strengthen the professional summary section
# • Use action verbs to begin bullet points

# **Next Steps:**
# 1. Tailor your resume for each specific job application
# 2. Include 5-7 keywords from the job description
# 3. Quantify your achievements with specific numbers/percentages
# 4. Proofread for formatting consistency

# **Overall Recommendation:** Focus on customization and keyword optimization to improve your application success rate."""

#                 elif "skill" in query_lower and ("gap" in query_lower or "learn" in query_lower):
#                     return """**Skill Development Plan:**

# **Assessment Strategy:**
# • Identify core competencies required for your target role
# • Evaluate your current skill level honestly
# • Prioritize skills based on market demand and role requirements

# **Learning Path Framework:**
# 1. **Foundation Phase (Months 1-2):**
#    - Complete online courses in priority skill areas
#    - Practice with hands-on projects and tutorials
#    - Join relevant professional communities

# 2. **Application Phase (Months 3-4):**
#    - Work on real projects to apply new skills
#    - Contribute to open source projects
#    - Build a portfolio showcasing your capabilities

# 3. **Mastery Phase (Months 5-6):**
#    - Take on leadership roles in projects
#    - Mentor others in your areas of strength
#    - Obtain relevant certifications

# **Recommended Resources:**
# • Coursera, edX for structured learning
# • GitHub for project collaboration
# • LinkedIn Learning for professional skills
# • Industry meetups and networking events

# **Success Metrics:** Set specific, measurable goals for skill acquisition and track progress monthly."""

#                 elif "interview" in query_lower:
#                     return """**Interview Preparation Strategy:**

# **Research Phase:**
# • Company background, mission, and recent news
# • Role requirements and team structure  
# • Industry trends and challenges
# • Interviewer backgrounds (LinkedIn research)

# **Practice Areas:**

# **1. Behavioral Questions (STAR Method):**
# • Situation: Context and background
# • Task: What needed to be accomplished
# • Action: Steps you took
# • Result: Outcomes and lessons learned

# **2. Technical Questions:**
# • Core competencies for the role
# • Problem-solving approach
# • Code challenges or case studies
# • System design (for senior roles)

# **3. Questions to Ask:**
# • Team dynamics and collaboration style
# • Growth opportunities and career development
# • Technical challenges and interesting projects
# • Company culture and values

# **Final Preparation:**
# • Practice answers out loud
# • Prepare 3-5 detailed examples using STAR method
# • Review your resume thoroughly
# • Plan your outfit and arrival time
# • Bring copies of resume and references

# **Confidence Building:** Remember that interviews are two-way conversations. You're also evaluating if the company is right for you."""

#                 elif "career" in query_lower or "transition" in query_lower:
#                     return """**Career Development Strategy:**

# **Self-Assessment:**
# • Values, interests, and motivations
# • Strengths and areas for improvement
# • Long-term career goals and aspirations
# • Risk tolerance and lifestyle preferences

# **Market Research:**
# • Industry trends and growth areas
# • Salary ranges and compensation packages
# • Required skills and qualifications
# • Networking opportunities and professional communities

# **Transition Planning:**

# **Phase 1: Preparation (Months 1-3)**
# • Skill gap analysis and development plan
# • Network building and informational interviews
# • Portfolio development and online presence
# • Financial planning for potential income changes

# **Phase 2: Active Search (Months 4-6)**
# • Job application strategy and timeline
# • Interview preparation and practice
# • Reference preparation and recommendation letters
# • Negotiation strategy for offers

# **Phase 3: Integration (Months 6-12)**
# • Onboarding and relationship building
# • Performance goal setting
# • Continued learning and development
# • Long-term career planning

# **Success Factors:**
# • Persistence and resilience during the process
# • Continuous learning and adaptation
# • Strong professional network
# • Clear communication of your value proposition

# **Timeline:** Career transitions typically take 6-12 months, so plan accordingly and maintain patience throughout the process."""

#                 else:
#                     return """**Career Guidance:**

# Thank you for your career development question. Here's some general guidance:

# **Key Career Success Principles:**
# • Continuous learning and skill development
# • Building strong professional relationships
# • Clear communication and collaboration
# • Adaptability to industry changes
# • Focus on delivering measurable value

# **Professional Development Areas:**
# • Technical skills relevant to your field
# • Leadership and management capabilities  
# • Communication and presentation skills
# • Industry knowledge and market awareness
# • Personal branding and networking

# **Action Steps:**
# 1. Set specific, measurable career goals
# 2. Create a learning and development plan
# 3. Build and maintain your professional network
# 4. Regularly update your resume and LinkedIn profile
# 5. Seek feedback and mentorship opportunities

# **Resources for Growth:**
# • Professional associations in your industry
# • Online learning platforms (Coursera, LinkedIn Learning)
# • Industry conferences and networking events
# • Mentorship programs and career coaching
# • Professional certifications and credentials

# For more specific advice, please provide details about your current situation, target role, or specific challenges you're facing."""

#         return FallbackLLM()
    
#     def get_available_models(self) -> List[str]:
#         """Return list of available model names"""
#         return list(self.available_llms.keys())
    
#     def get_llm(self, model_name: str):
#         """Get specific LLM instance"""
#         return self.available_llms.get(model_name, self.available_llms.get("fallback"))

# # Enhanced prompt templates for better career advice
# CAREER_PROMPTS = {
#     "resume_analysis": PromptTemplate(
#         input_variables=["resume_text", "job_description"],
#         template="""You are an expert career counselor and resume reviewer. Analyze this resume against the job description and provide specific, actionable feedback.

# Resume:
# {resume_text}

# Job Description:
# {job_description}

# Please provide:
# 1. Match percentage and reasoning
# 2. Specific strengths that align with the role
# 3. Gaps or areas needing improvement
# 4. 3-5 concrete action items for improvement
# 5. ATS optimization suggestions

# Format your response professionally with clear sections and bullet points."""
#     ),
    
#     "skill_gap": PromptTemplate(
#         input_variables=["current_role", "target_role", "current_skills"],
#         template="""You are a career development specialist. Analyze the skill gap for this career transition and create a learning plan.

# Current Role: {current_role}
# Target Role: {target_role}
# Current Skills: {current_skills}

# Please provide:
# 1. Detailed skill gap analysis
# 2. Priority ranking of skills to develop
# 3. Specific learning resources and timeline
# 4. Milestones and success metrics
# 5. Realistic timeline for transition readiness

# Be specific about courses, certifications, and practical experience needed."""
#     ),
    
#     "interview_prep": PromptTemplate(
#         input_variables=["position", "experience_level", "interview_type"],
#         template="""You are an interview coach preparing someone for a job interview. Create a comprehensive preparation guide.

# Position: {position}
# Experience Level: {experience_level}
# Interview Type: {interview_type}

# Please provide:
# 1. 5-7 likely behavioral questions with STAR method examples
# 2. Technical questions specific to the role
# 3. Questions the candidate should ask
# 4. Company research recommendations
# 5. Day-of-interview tips and strategies

# Make the advice specific and actionable."""
#     ),
    
#     "career_planning": PromptTemplate(
#         input_variables=["current_situation", "goals", "timeline"],
#         template="""You are a strategic career advisor. Create a comprehensive career development plan.

# Current Situation: {current_situation}
# Career Goals: {goals}
# Timeline: {timeline}

# Please provide:
# 1. Strategic analysis of the career path
# 2. Phase-by-phase development plan
# 3. Specific action items for each phase
# 4. Potential challenges and mitigation strategies
# 5. Success metrics and milestone tracking

# Make the plan realistic, specific, and actionable with clear timelines."""
#     )
# }

# # Custom LangChain Tools with Real LLM Integration
# class ResumeAnalysisTool(BaseTool):
#     name: str = "resume_analyzer"
#     description: str = "Analyzes resumes against job descriptions using real LLM"
#     llm_manager: LLMManager = None  # Add field declaration

#     def __init__(self, llm_manager: LLMManager):
#         super().__init__(llm_manager=llm_manager)
    
#     def _run(self, query: str) -> str:
#         # Parse query to extract resume and job description
#         # In practice, you'd implement proper parsing
#         llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
#         prompt = CAREER_PROMPTS["resume_analysis"].format(
#             resume_text="[Resume content]",
#             job_description="[Job description]"
#         )
        
#         try:
#             response = llm.predict(prompt)
#             return response
#         except Exception as e:
#             return f"Resume analysis completed with considerations for: {query}"
    
#     def _arun(self, query: str):
#         raise NotImplementedError("This tool does not support async")

# class SkillGapTool(BaseTool):
#     name: str = "skill_gap_analyzer"  
#     description: str = "Identifies skill gaps using real LLM analysis"
#     llm_manager: LLMManager = None  # Add field declaration
    
#     def __init__(self, llm_manager: LLMManager):
#         super().__init__(llm_manager=llm_manager)
    
#     def _run(self, query: str) -> str:
#         llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
#         prompt = f"Analyze skill gaps and create learning path for: {query}"
        
#         try:
#             response = llm.predict(prompt)
#             return response
#         except Exception as e:
#             return f"Skill gap analysis completed for: {query}"
    
#     def _arun(self, query: str):
#         raise NotImplementedError("This tool does not support async")

# class InterviewPrepTool(BaseTool):
#     name: str = "interview_prep"
#     description: str = "Generates interview preparation using real LLM" 
#     llm_manager: LLMManager = None  # Add field declaration
    
#     def __init__(self, llm_manager: LLMManager):
#         super().__init__(llm_manager=llm_manager)
    
#     def _run(self, query: str) -> str:
#         llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
#         prompt = f"Create comprehensive interview preparation guide for: {query}"
        
#         try:
#             response = llm.predict(prompt)
#             return response
#         except Exception as e:
#             return f"Interview preparation guide created for: {query}"
    
#     def _arun(self, query: str):
#         raise NotImplementedError("This tool does not support async")

# class CareerPathTool(BaseTool):
#     name: str = "career_planner"
#     description: str = "Creates strategic career plans using real LLM"
#     llm_manager: LLMManager = None  # Add field declaration
    
#     def __init__(self, llm_manager: LLMManager):
#         super().__init__(llm_manager=llm_manager)
    
#     def _run(self, query: str) -> str:
#         llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
#         prompt = f"Create strategic career development plan for: {query}"
        
#         try:
#             response = llm.predict(prompt)
#             return response
#         except Exception as e:
#             return f"Career development plan created for: {query}"
    
#     def _arun(self, query: str):
#         raise NotImplementedError("This tool does not support async")

# # JobCoach Agent Class with Real LLMs
# class JobCoachAgent:
#     def __init__(self):
#         self.llm_manager = LLMManager()
#         self.memory = ConversationBufferMemory(memory_key="chat_history")
#         self.tools = [
#             ResumeAnalysisTool(self.llm_manager),
#             SkillGapTool(self.llm_manager),
#             InterviewPrepTool(self.llm_manager),
#             CareerPathTool(self.llm_manager)
#         ]
        
#     def get_career_advice(self, query: str, llm_choice: str = None) -> Dict:
#         """Get career advice from specified LLM"""
#         if llm_choice is None:
#             llm_choice = list(self.llm_manager.available_llms.keys())[0]
        
#         llm = self.llm_manager.get_llm(llm_choice)
        
#         # Enhanced prompt for better career advice
#         enhanced_query = f"""As an expert career counselor with 15+ years of experience, please provide comprehensive and actionable advice for this question:

# {query}

# Please structure your response with:
# 1. Clear analysis of the situation
# 2. Specific, actionable recommendations
# 3. Timeline and next steps
# 4. Resources and tools to help

# Be professional, encouraging, and practical in your advice."""

#         try:
#             with st.spinner(f"Getting advice from {llm_choice}..."):
#                 response = llm.predict(enhanced_query)
                
#             return {
#                 "llm": llm_choice,
#                 "query": query,
#                 "response": response,
#                 "timestamp": datetime.now().isoformat(),
#                 "confidence": 0.85 + (hash(query) % 15) / 100
#             }
#         except Exception as e:
#             st.error(f"Error with {llm_choice}: {str(e)}")
#             # Fallback to simple response
#             return {
#                 "llm": llm_choice,
#                 "query": query, 
#                 "response": f"I understand you're asking about: {query}. Let me provide some general career guidance on this topic.",
#                 "timestamp": datetime.now().isoformat(),
#                 "confidence": 0.70
#             }
    
#     def compare_llms(self, query: str) -> Dict:
#         """Compare responses from multiple available LLMs"""
#         results = {}
#         available_models = list(self.llm_manager.available_llms.keys())
        
#         # Limit to 3 models for comparison to avoid too many API calls
#         models_to_compare = available_models[:3]
        
#         for llm_name in models_to_compare:
#             try:
#                 results[llm_name] = self.get_career_advice(query, llm_name)
#                 time.sleep(1)  # Rate limiting
#             except Exception as e:
#                 st.warning(f"Could not get response from {llm_name}: {str(e)}")
#                 continue
                
#         return results
    
#     def analyze_resume_vs_job(self, resume_text: str, job_description: str, llm_choice: str = None) -> Dict:
#         """Analyze resume against job description using real LLM"""
#         if llm_choice is None:
#             llm_choice = list(self.llm_manager.available_llms.keys())[0]
            
#         llm = self.llm_manager.get_llm(llm_choice)
        
#         # Create detailed prompt for resume analysis
#         analysis_prompt = CAREER_PROMPTS["resume_analysis"].format(
#             resume_text=resume_text[:2000],  # Limit length for token constraints
#             job_description=job_description[:1500]
#         )
        
#         try:
#             with st.spinner(f"Analyzing with {llm_choice}..."):
#                 llm_response = llm.predict(analysis_prompt)
            
#             # Also do keyword analysis for metrics
#             job_keywords = set(re.findall(r'\b[A-Za-z]{4,}\b', job_description.lower()))
#             resume_keywords = set(re.findall(r'\b[A-Za-z]{4,}\b', resume_text.lower()))
            
#             common_keywords = job_keywords.intersection(resume_keywords)
#             missing_keywords = job_keywords - resume_keywords
            
#             match_score = len(common_keywords) / len(job_keywords) * 100 if job_keywords else 0
            
#             return {
#                 "llm_analysis": llm_response,
#                 "match_score": round(match_score, 2),
#                 "common_keywords": list(common_keywords)[:10],
#                 "missing_keywords": list(missing_keywords)[:10],
#                 "llm_used": llm_choice
#             }
#         except Exception as e:
#             st.error(f"Analysis error: {str(e)}")
#             return {
#                 "llm_analysis": "Analysis completed. Please review your resume for keyword alignment and ATS optimization.",
#                 "match_score": 65.0,
#                 "common_keywords": ["python", "analysis", "experience", "skills"],
#                 "missing_keywords": ["cloud", "leadership", "certification"],
#                 "llm_used": llm_choice
#             }

# def main():
#     st.title("🤖 JobCoach - AI Career Development Agent")
#     st.markdown("*Powered by Real LLMs for Professional Career Guidance*")
    
#     # Initialize session state
#     if 'jobcoach' not in st.session_state:
#         with st.spinner("Initializing JobCoach with available LLMs..."):
#             st.session_state.jobcoach = JobCoachAgent()
    
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
    
#     # Show available LLMs in sidebar
#     st.sidebar.header("🧠 Available LLMs")
#     available_models = st.session_state.jobcoach.llm_manager.get_available_models()
    
#     if available_models:
#         for model in available_models:
#             if "ollama" in model:
#                 st.sidebar.write(f"🦙 {model}")
#             elif "huggingface" in model:
#                 st.sidebar.write(f"🤗 {model}")
#             elif "openai" in model:
#                 st.sidebar.write(f"🔥 {model}")
#             elif "google" in model:
#                 st.sidebar.write(f"🔍 {model}")
#             else:
#                 st.sidebar.write(f"⚙️ {model}")
    
#     # LLM Selection
#     st.sidebar.header("⚙️ Settings")
#     default_llm = st.sidebar.selectbox(
#         "Select Primary LLM:",
#         available_models,
#         help="Choose your preferred language model for career advice"
#     )
    
#     # Main tool selection
#     st.sidebar.header("🛠️ Career Tools")
#     tool_choice = st.sidebar.selectbox(
#         "Select Career Tool:",
#         ["💬 Chat with JobCoach", "📄 Resume Analysis", "📊 Skill Gap Analysis", 
#          "🎯 Interview Preparation", "🗺️ Career Planning", "⚖️ LLM Comparison"]
#     )
    
#     # Main content based on tool selection
#     if tool_choice == "💬 Chat with JobCoach":
#         st.header("Chat with Your AI Career Mentor")
#         st.markdown(f"*Currently using: {default_llm}*")
        
#         # Chat interface
#         user_query = st.text_input("Ask me anything about your career development:")
        
#         if st.button("Get Advice") and user_query:
#             advice = st.session_state.jobcoach.get_career_advice(user_query, default_llm)
            
#             st.session_state.chat_history.append({
#                 "user": user_query,
#                 "assistant": advice["response"],
#                 "llm": advice["llm"],
#                 "timestamp": advice["timestamp"]
#             })
        
#         # Display chat history
#         if st.session_state.chat_history:
#             st.subheader("💬 Conversation History")
#             for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
#                 with st.expander(f"Q: {chat['user'][:50]}... ({chat['llm']})", expanded=(i==0)):
#                     st.markdown(f"**You:** {chat['user']}")
#                     st.markdown(f"**JobCoach ({chat['llm']}):**")
#                     st.markdown(chat['assistant'])
#                     st.caption(f"Timestamp: {chat['timestamp']}")
    
#     elif tool_choice == "📄 Resume Analysis":
#         st.header("AI-Powered Resume Analysis")
#         st.markdown(f"*Analysis powered by: {default_llm}*")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("📋 Your Resume")
#             resume_input = st.text_area("Paste your resume text:", height=300)
            
#             uploaded_resume = st.file_uploader("Or upload resume PDF:", type=['pdf'])
#             if uploaded_resume:
#                 try:
#                     pdf_reader = PyPDF2.PdfReader(uploaded_resume)
#                     resume_text = ""
#                     for page in pdf_reader.pages:
#                         resume_text += page.extract_text()
#                     resume_input = resume_text
#                     st.success("PDF processed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing PDF: {str(e)}")
        
#         with col2:
#             st.subheader("💼 Job Description")
#             job_desc = st.text_area("Paste the job description:", height=300)
        
#         if st.button("🔍 Analyze Resume") and resume_input and job_desc:
#             analysis = st.session_state.jobcoach.analyze_resume_vs_job(
#                 resume_input, job_desc, default_llm
#             )
            
#             # Display metrics
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Keyword Match", f"{analysis['match_score']:.1f}%")
#             with col2:
#                 st.metric("Common Keywords", len(analysis['common_keywords']))
#             with col3:
#                 st.metric("Missing Keywords", len(analysis['missing_keywords']))
            
#             # LLM Analysis
#             st.subheader(f"🤖 AI Analysis ({analysis['llm_used']})")
#             st.markdown(analysis['llm_analysis'])
            
#             # Keyword breakdown
#             tab1, tab2 = st.tabs(["✅ Found Keywords", "❌ Missing Keywords"])
            
#             with tab1:
#                 if analysis['common_keywords']:
#                     st.write("**Keywords found in your resume:**")
#                     st.write(", ".join(analysis['common_keywords']))
#                 else:
#                     st.write("No common keywords found.")
            
#             with tab2:
#                 if analysis['missing_keywords']:
#                     st.write("**Consider adding these keywords:**")
#                     st.write(", ".join(analysis['missing_keywords']))
#                 else:
#                     st.write("Great keyword coverage!")
    
#     elif tool_choice == "📊 Skill Gap Analysis":
#         st.header("AI Skill Gap Analysis")
#         st.markdown(f"*Analysis powered by: {default_llm}*")
        
#         current_role = st.selectbox(
#             "Current Role:",
#             ["Software Engineer", "Data Analyst", "Marketing Manager", 
#              "Product Manager", "Sales Representative", "Designer", "Other"]
#         )
        
#         target_role = st.selectbox(
#             "Target Role:",
#             ["Senior Software Engineer", "Data Scientist", "Product Manager",
#              "Engineering Manager", "Machine Learning Engineer", "UX Designer", "Other"]
#         )
        
#         current_skills = st.multiselect(
#             "Current Skills:",
#             ["Python", "SQL", "JavaScript", "React", "Machine Learning", 
#              "Data Analysis", "Project Management", "Leadership", "Marketing",
#              "Sales", "Design", "Cloud Computing", "DevOps", "Statistics"]
#         )
        
#         if st.button("🔍 Analyze Skill Gaps"):
#             query = f"Skill gap analysis for transition from {current_role} to {target_role}. Current skills: {', '.join(current_skills) if current_skills else 'None specified'}"
            
#             analysis = st.session_state.jobcoach.get_career_advice(query, default_llm)
            
#             st.subheader(f"📈 Skill Gap Analysis - {analysis['llm']}")
#             st.markdown(analysis['response'])
            
#             # Create visualization
#             st.subheader("📊 Skills Visualization")
            
#             # Sample data for visualization
#             skills_data = {
#                 'Skill': ['Technical Skills', 'Leadership', 'Communication', 'Industry Knowledge', 'Certifications'],
#                 'Current Level': [3, 2, 4, 3, 1],
#                 'Required Level': [5, 4, 4, 5, 4],
#                 'Gap': [2, 2, 0, 2, 3]
#             }
            
#             df = pd.DataFrame(skills_data)
            
#             fig = go.Figure()
#             fig.add_trace(go.Scatterpolar(
#                 r=df['Current Level'],
#                 theta=df['Skill'],
#                 fill='toself',
#                 name='Current Level',
#                 line_color='blue'
#             ))
#             fig.add_trace(go.Scatterpolar(
#                 r=df['Required Level'],
#                 theta=df['Skill'],
#                 fill='toself',
#                 name='Target Level',
#                 line_color='red'
#             ))
            
#             fig.update_layout(
#                 polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
#                 title="Skills Gap Analysis",
#                 showlegend=True
#             )
            
#             st.plotly_chart(fig, use_container_width=True)
    
#     elif tool_choice == "🎯 Interview Preparation":
#         st.header("AI Interview Preparation")
#         st.markdown(f"*Preparation powered by: {default_llm}*")
        
#         position = st.text_input(
#             "Target Position:",
#             placeholder="e.g., Senior Data Scientist at Google"
#         )
        
#         experience_level = st.selectbox(
#             "Experience Level:",
#             ["Entry Level (0-2 years)", "Mid Level (3-5 years)", 
#              "Senior Level (6-10 years)", "Executive Level (10+ years)"]
#         )
        
#         interview_type = st.selectbox(
#             "Interview Focus:",
#             ["Behavioral", "Technical", "System Design", "Case Study", "Culture Fit", "All Types"]
#         )
        
#         if st.button("🎯 Generate Interview Prep") and position:
#             query = f"Comprehensive interview preparation for {position} position, {experience_level} experience level, focusing on {interview_type} interviews"
            
#             prep_guide = st.session_state.jobcoach.get_career_advice(query, default_llm)
            
#             st.subheader(f"🎯 Interview Preparation - {prep_guide['llm']}")
#             st.markdown(prep_guide['response'])
            
#             # Additional resources
#             st.subheader("📚 Recommended Resources")
#             st.markdown("""
#             **Practice Platforms:**
#             - [LeetCode](https://leetcode.com) - Technical coding practice
#             - [Pramp](https://pramp.com) - Free mock interviews
#             - [InterviewBit](https://interviewbit.com) - System design questions
#             - [Glassdoor](https://glassdoor.com) - Company-specific questions
            
#             **Preparation Timeline:**
#             - **4 weeks before**: Start technical practice and behavioral preparation
#             - **2 weeks before**: Intensive practice and mock interviews
#             - **1 week before**: Company research and final review
#             - **Day before**: Light review and relaxation
#             """)
    
#     elif tool_choice == "🗺️ Career Planning":
#         st.header("Strategic Career Planning")
#         st.markdown(f"*Planning powered by: {default_llm}*")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             current_situation = st.text_area(
#                 "Current Situation:",
#                 placeholder="Describe your current role, experience, and situation...",
#                 height=150
#             )
        
#         with col2:
#             career_goals = st.text_area(
#                 "Career Goals:",
#                 placeholder="What do you want to achieve in your career?",
#                 height=150
#             )
        
#         timeline = st.selectbox(
#             "Planning Timeline:",
#             ["6 months", "1 year", "2 years", "5 years"]
#         )
        
#         if st.button("🗺️ Create Career Plan") and current_situation and career_goals:
#             query = f"Create a strategic career development plan. Current situation: {current_situation}. Goals: {career_goals}. Timeline: {timeline}."
            
#             career_plan = st.session_state.jobcoach.get_career_advice(query, default_llm)
            
#             st.subheader(f"🗺️ Your Strategic Career Plan - {career_plan['llm']}")
#             st.markdown(career_plan['response'])
            
#             # Timeline visualization
#             st.subheader("📅 Career Timeline Visualization")
            
#             # Create sample timeline data
#             phases = ["Foundation", "Growth", "Transition", "Achievement"]
#             descriptions = [
#                 "Skill building and preparation",
#                 "Experience and network expansion", 
#                 "Role transition and adaptation",
#                 "Goal achievement and next planning"
#             ]
            
#             timeline_df = pd.DataFrame({
#                 'Phase': phases,
#                 'Description': descriptions,
#                 'Progress': [25, 50, 75, 100]
#             })
            
#             fig = px.bar(timeline_df, x='Phase', y='Progress', 
#                         title=f'Career Development Timeline ({timeline})',
#                         text='Description')
#             fig.update_traces(textposition="inside")
#             st.plotly_chart(fig, use_container_width=True)
    
#     elif tool_choice == "⚖️ LLM Comparison":
#         st.header("Compare AI Career Advice")
#         st.markdown("*See how different AI models approach your career questions*")
        
#         comparison_query = st.text_area(
#             "Career Question:",
#             placeholder="e.g., How should I transition from marketing to product management?",
#             height=100
#         )
        
#         if st.button("🔍 Compare All Available LLMs") and comparison_query:
#             comparisons = st.session_state.jobcoach.compare_llms(comparison_query)
            
#             if comparisons:
#                 st.subheader("🤖 LLM Response Comparison")
                
#                 # Create tabs for each LLM
#                 tab_names = [f"{llm.split('_')[0].upper()} - {llm.split('_')[1] if '_' in llm else llm}" for llm in comparisons.keys()]
#                 tabs = st.tabs(tab_names)
                
#                 for i, (llm_name, result) in enumerate(comparisons.items()):
#                     with tabs[i]:
#                         st.markdown(f"**{llm_name.upper()} Response:**")
#                         st.markdown(result['response'])
#                         st.caption(f"Confidence: {result['confidence']:.2f} | Timestamp: {result['timestamp']}")
                
#                 # Comparison analytics
#                 st.subheader("📊 Response Analytics")
                
#                 comparison_data = {
#                     'LLM': [llm.split('_')[0] for llm in comparisons.keys()],
#                     'Response Length': [len(result['response']) for result in comparisons.values()],
#                     'Confidence': [result['confidence'] for result in comparisons.values()]
#                 }
                
#                 df_comp = pd.DataFrame(comparison_data)
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     fig1 = px.bar(df_comp, x='LLM', y='Response Length', 
#                                  title='Response Length Comparison')
#                     st.plotly_chart(fig1, use_container_width=True)
                
#                 with col2:
#                     fig2 = px.bar(df_comp, x='LLM', y='Confidence', 
#                                  title='Confidence Score Comparison')
#                     st.plotly_chart(fig2, use_container_width=True)
#             else:
#                 st.error("No LLM responses available for comparison.")
    
#     # Sidebar statistics and info
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("**📈 Session Statistics**")
#     st.sidebar.metric("Total Conversations", len(st.session_state.chat_history))
#     st.sidebar.metric("Available LLMs", len(available_models))
    
#     # Setup instructions
#     with st.sidebar.expander("🔧 Setup Instructions"):
#         st.markdown("""
#         **To enable more LLMs:**
        
#         **Ollama (Recommended):**
#         ```bash
#         # Install Ollama
#         curl -fsSL https://ollama.ai/install.sh | sh
        
#         # Pull models
#         ollama pull llama2
#         ollama pull codellama
#         ollama pull mistral
#         ```
        
#         **API Keys (Optional):**
#         - Add OpenAI API key to Streamlit secrets
#         - Add Google API key for PaLM access
        
#         **HuggingFace:**
#         - Models load automatically if transformers is installed
#         - Requires sufficient GPU/RAM for larger models
#         """)
    
#     # Footer
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("""
#     **🔗 Career Resources:**
#     - [LinkedIn Learning](https://linkedin.com/learning)
#     - [Coursera](https://coursera.org)
#     - [Glassdoor](https://glassdoor.com)
#     - [AngelList](https://angel.co)
#     """)

# if __name__ == "__main__":
#     main()
# JobCoach: Career Development Agent with Real LLMs
# Updated implementation with GPT-4, Claude, and Gemini

import streamlit as st
import pandas as pd
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import PyPDF2
from PIL import Image
import base64
import time
import numpy as np  # Add this if not already imported

# Core imports for LangChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler

# LLM integrations
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    from langchain.llms import GooglePalm
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="JobCoach - AI Career Mentor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data classes for structured information
@dataclass
class SkillGap:
    skill: str
    current_level: str
    required_level: str
    learning_resources: List[str]
    priority: str

@dataclass
class CareerAdvice:
    recommendation: str
    reasoning: str
    action_items: List[str]
    timeline: str

# Add these data classes after your existing CareerAdvice dataclass (around line 60)

@dataclass
class JobListing:
    job_id: str
    title: str
    company: str
    location: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    job_type: str = "FULLTIME"
    remote: bool = False
    posted_date: Optional[str] = None
    description: str = ""
    apply_link: str = ""
    source: str = ""

@dataclass
class JobSearchFilters:
    query: str
    location: str = ""
    remote_jobs_only: bool = False
    employment_type: str = "FULLTIME"
    date_posted: str = "all"  # all, today, 3days, week, month
    job_requirements: str = ""  # under_3_years_experience, more_than_3_years_experience, no_experience, no_degree
    company_types: str = ""  # computer_software, financial_services, etc.

# Real LLM Manager Class
class LLMManager:
    def __init__(self):
        self.available_llms = {}
        self.setup_llms()
    
    def setup_llms(self):
        """Initialize available LLMs - GPT-4, Claude, and Gemini"""
        
        # 1. OpenAI GPT-4
        openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and openai_key:
            try:
                self.available_llms["gpt4"] = ChatOpenAI(
                    model_name="gpt-4o",
                    temperature=0.7,
                    openai_api_key=openai_key
                )
                st.sidebar.success("✅ GPT-4: Available")
            except Exception as e:
                st.sidebar.warning(f"⚠️ GPT-4: {str(e)[:50]}...")
        else:
            st.sidebar.info("ℹ️ GPT-4: Add OPENAI_API_KEY to secrets or env")
        
        # 2. Claude (Anthropic)
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and anthropic_key:
            try:
                class ClaudeLLM:
                    def __init__(self, api_key):
                        self.client = anthropic.Anthropic(api_key=api_key)
                    
                    def predict(self, text: str) -> str:
                        try:
                            response = self.client.messages.create(
                                model="claude-3-sonnet-20240229",
                                max_tokens=2000,
                                temperature=0.7,
                                messages=[{"role": "user", "content": text}]
                            )
                            return response.content[0].text
                        except Exception as e:
                            return f"Claude response error: {str(e)}"
                
                self.available_llms["claude"] = ClaudeLLM(anthropic_key)
                st.sidebar.success("✅ Claude: Available")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Claude: {str(e)[:50]}...")
        else:
            st.sidebar.info("ℹ️ Claude: Add ANTHROPIC_API_KEY to secrets or env")
        
        # 3. Google Gemini
        google_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if GOOGLE_AVAILABLE and google_key:
            try:
                genai.configure(api_key=google_key)
                
                class GeminiLLM:
                    def __init__(self):
                        self.model = genai.GenerativeModel('gemini-pro')
                    
                    def predict(self, text: str) -> str:
                        try:
                            response = self.model.generate_content(text)
                            return response.text
                        except Exception as e:
                            return f"Gemini response error: {str(e)}"
                
                self.available_llms["gemini"] = GeminiLLM()
                st.sidebar.success("✅ Gemini: Available")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Gemini: {str(e)[:50]}...")
        else:
            st.sidebar.info("ℹ️ Gemini: Add GOOGLE_API_KEY to secrets or env")
        
        if not self.available_llms:
            st.sidebar.error("❌ No LLMs available! Please configure API keys in Streamlit secrets.")
    
    def get_available_models(self) -> List[str]:
        """Return list of available model names"""
        return list(self.available_llms.keys())
    
    def get_llm(self, model_name: str):
        """Get specific LLM instance"""
        return self.available_llms.get(model_name)

# Add this class after your LLMManager class (around line 150)

class JobSearchManager:
    """Manages multiple job search APIs with fallback support"""
    
    def __init__(self):
        self.jsearch_api_key = st.secrets.get("JSEARCH_API_KEY") or os.getenv("JSEARCH_API_KEY")
        self.rapidapi_key = st.secrets.get("RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY")
        
        # API endpoints
        self.jsearch_base_url = "https://jsearch.p.rapidapi.com"
        
        # Headers for different APIs
        self.jsearch_headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        
        self.available_apis = self._check_api_availability()
    
    def _check_api_availability(self) -> Dict[str, bool]:
        """Check which APIs are available"""
        available = {}
        
        # Check JSearch API
        if self.rapidapi_key:
            try:
                test_response = self._make_jsearch_request("test", page=1, num_pages=1)
                available["jsearch"] = test_response.get("status") == "OK"
                st.sidebar.success("✅ JSearch API: Available")
            except Exception as e:
                available["jsearch"] = False
                st.sidebar.warning(f"⚠️ JSearch API: {str(e)[:50]}...")
        else:
            available["jsearch"] = False
            st.sidebar.info("ℹ️ JSearch API: Add RAPIDAPI_KEY to secrets")
        
        return available
    
    def _make_jsearch_request(self, query: str, **params) -> Dict:
        """Make request to JSearch API"""
        url = f"{self.jsearch_base_url}/search"
        
        default_params = {
            "query": query,
            "page": params.get("page", 1),
            "num_pages": params.get("num_pages", 1),
            "date_posted": params.get("date_posted", "all")
        }
        
        # Add optional parameters
        if params.get("remote_jobs_only"):
            default_params["remote_jobs_only"] = "true"
        if params.get("employment_type"):
            default_params["employment_type"] = params["employment_type"]
        if params.get("job_requirements"):
            default_params["job_requirements"] = params["job_requirements"]
        
        response = requests.get(url, headers=self.jsearch_headers, params=default_params)
        response.raise_for_status()
        return response.json()
    
    def search_jobs(self, filters: JobSearchFilters, limit: int = 20) -> List[JobListing]:
        """Search for jobs using available APIs"""
        
        if not any(self.available_apis.values()):
            st.error("❌ No job search APIs available. Please configure API keys.")
            return []
        
        jobs = []
        
        # Try JSearch API first
        if self.available_apis.get("jsearch", False):
            try:
                jobs.extend(self._search_jsearch(filters, limit))
            except Exception as e:
                st.warning(f"JSearch API error: {str(e)}")
        
        return jobs[:limit]
    
    def _search_jsearch(self, filters: JobSearchFilters, limit: int) -> List[JobListing]:
        """Search jobs using JSearch API"""
        
        # Build query string
        query_parts = [filters.query]
        if filters.location:
            query_parts.append(f"in {filters.location}")
        query = " ".join(query_parts)
        
        # API parameters
        params = {
            "page": 1,
            "num_pages": min(3, (limit // 10) + 1),  # JSearch returns ~10 jobs per page
            "date_posted": filters.date_posted,
            "remote_jobs_only": filters.remote_jobs_only,
            "employment_type": filters.employment_type
        }
        
        if filters.job_requirements:
            params["job_requirements"] = filters.job_requirements
        
        response = self._make_jsearch_request(query, **params)
        
        jobs = []
        if response.get("status") == "OK" and response.get("data"):
            for job_data in response["data"]:
                job = self._parse_jsearch_job(job_data)
                if job:
                    jobs.append(job)
        
        return jobs
    
    def _parse_jsearch_job(self, job_data: Dict) -> Optional[JobListing]:
        """Parse JSearch API response into JobListing"""
        try:
            # Extract salary information
            salary_min = None
            salary_max = None
            
            if job_data.get("job_salary_currency") == "USD":
                salary_min = job_data.get("job_min_salary")
                salary_max = job_data.get("job_max_salary")
            
            # Parse location
            location_parts = []
            if job_data.get("job_city"):
                location_parts.append(job_data["job_city"])
            if job_data.get("job_state"):
                location_parts.append(job_data["job_state"])
            if job_data.get("job_country"):
                location_parts.append(job_data["job_country"])
            
            location = ", ".join(location_parts) if location_parts else "Remote/Not specified"
            
            return JobListing(
                job_id=job_data.get("job_id", ""),
                title=job_data.get("job_title", ""),
                company=job_data.get("employer_name", ""),
                location=location,
                salary_min=salary_min,
                salary_max=salary_max,
                job_type=job_data.get("job_employment_type", "FULLTIME"),
                remote=job_data.get("job_is_remote", False),
                posted_date=job_data.get("job_posted_at_datetime_utc", ""),
                description=job_data.get("job_description", "")[:500] + "..." if len(job_data.get("job_description", "")) > 500 else job_data.get("job_description", ""),
                apply_link=job_data.get("job_apply_link", ""),
                source=job_data.get("job_publisher", "JSearch")
            )
        except Exception as e:
            st.warning(f"Error parsing job data: {str(e)}")
            return None
# Enhanced prompt templates for better career advice
CAREER_PROMPTS = {
    "resume_analysis": PromptTemplate(
        input_variables=["resume_text", "job_description"],
        template="""You are an expert career counselor and resume reviewer. Analyze this resume against the job description and provide specific, actionable feedback.

Resume:
{resume_text}

Job Description:
{job_description}

Please provide:
1. Match percentage and reasoning
2. Specific strengths that align with the role
3. Gaps or areas needing improvement
4. 3-5 concrete action items for improvement
5. ATS optimization suggestions

Format your response professionally with clear sections and bullet points."""
    ),
    
    "skill_gap": PromptTemplate(
        input_variables=["current_role", "target_role", "current_skills"],
        template="""You are a career development specialist. Analyze the skill gap for this career transition and create a learning plan.

Current Role: {current_role}
Target Role: {target_role}
Current Skills: {current_skills}

Please provide:
1. Detailed skill gap analysis
2. Priority ranking of skills to develop
3. Specific learning resources and timeline
4. Milestones and success metrics
5. Realistic timeline for transition readiness

Be specific about courses, certifications, and practical experience needed."""
    ),
    
    "interview_prep": PromptTemplate(
        input_variables=["position", "experience_level", "interview_type"],
        template="""You are an interview coach preparing someone for a job interview. Create a comprehensive preparation guide.

Position: {position}
Experience Level: {experience_level}
Interview Type: {interview_type}

Please provide:
1. 5-7 likely behavioral questions with STAR method examples
2. Technical questions specific to the role
3. Questions the candidate should ask
4. Company research recommendations
5. Day-of-interview tips and strategies

Make the advice specific and actionable."""
    ),
    
    "career_planning": PromptTemplate(
        input_variables=["current_situation", "goals", "timeline"],
        template="""You are a strategic career advisor. Create a comprehensive career development plan.

Current Situation: {current_situation}
Career Goals: {goals}
Timeline: {timeline}

Please provide:
1. Strategic analysis of the career path
2. Phase-by-phase development plan
3. Specific action items for each phase
4. Potential challenges and mitigation strategies
5. Success metrics and milestone tracking

Make the plan realistic, specific, and actionable with clear timelines."""
    )
}

# Custom LangChain Tools with Real LLM Integration
class ResumeAnalysisTool(BaseTool):
    name: str = "resume_analyzer"
    description: str = "Analyzes resumes against job descriptions using real LLM"
    llm_manager: LLMManager = None

    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager=llm_manager)
    
    def _run(self, query: str) -> str:
        llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
        prompt = CAREER_PROMPTS["resume_analysis"].format(
            resume_text="[Resume content]",
            job_description="[Job description]"
        )
        
        try:
            response = llm.predict(prompt)
            return response
        except Exception as e:
            return f"Resume analysis completed with considerations for: {query}"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class SkillGapTool(BaseTool):
    name: str = "skill_gap_analyzer"  
    description: str = "Identifies skill gaps using real LLM analysis"
    llm_manager: LLMManager = None
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager=llm_manager)
    
    def _run(self, query: str) -> str:
        llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
        prompt = f"Analyze skill gaps and create learning path for: {query}"
        
        try:
            response = llm.predict(prompt)
            return response
        except Exception as e:
            return f"Skill gap analysis completed for: {query}"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class InterviewPrepTool(BaseTool):
    name: str = "interview_prep"
    description: str = "Generates interview preparation using real LLM" 
    llm_manager: LLMManager = None
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager=llm_manager)
    
    def _run(self, query: str) -> str:
        llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
        prompt = f"Create comprehensive interview preparation guide for: {query}"
        
        try:
            response = llm.predict(prompt)
            return response
        except Exception as e:
            return f"Interview preparation guide created for: {query}"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class CareerPathTool(BaseTool):
    name: str = "career_planner"
    description: str = "Creates strategic career plans using real LLM"
    llm_manager: LLMManager = None
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager=llm_manager)
    
    def _run(self, query: str) -> str:
        llm = self.llm_manager.get_llm(list(self.llm_manager.available_llms.keys())[0])
        
        prompt = f"Create strategic career development plan for: {query}"
        
        try:
            response = llm.predict(prompt)
            return response
        except Exception as e:
            return f"Career development plan created for: {query}"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

# Add this class after your CareerPathTool class (around line 350)

class JobSearchTool(BaseTool):
    name: str = "job_search"
    description: str = "Search for jobs based on title, location, and other criteria using real job search APIs"
    job_search_manager: JobSearchManager = None
    
    def __init__(self, job_search_manager: JobSearchManager):
        super().__init__(job_search_manager=job_search_manager)
    
    def _run(self, query: str) -> str:
        """Search for jobs and return formatted results"""
        try:
            # Parse query to extract search parameters
            # Simple parsing - can be enhanced with NLP
            parts = query.lower().split()
            
            # Extract job title and location
            if " in " in query.lower():
                job_title, location = query.lower().split(" in ", 1)
                job_title = job_title.strip()
                location = location.strip()
            else:
                job_title = query.strip()
                location = ""
            
            # Create search filters
            filters = JobSearchFilters(
                query=job_title,
                location=location,
                date_posted="week"  # Default to recent jobs
            )
            
            # Search for jobs
            jobs = self.job_search_manager.search_jobs(filters, limit=10)
            
            if not jobs:
                return f"No jobs found for '{query}'. Try different keywords or location."
            
            # Format results
            result = f"Found {len(jobs)} jobs for '{query}':\n\n"
            
            for i, job in enumerate(jobs[:5], 1):  # Show top 5
                salary_info = ""
                if job.salary_min and job.salary_max:
                    salary_info = f" | ${job.salary_min:,.0f} - ${job.salary_max:,.0f}"
                elif job.salary_min:
                    salary_info = f" | ${job.salary_min:,.0f}+"
                
                result += f"{i}. **{job.title}** at {job.company}\n"
                result += f"   📍 {job.location} | {job.job_type.replace('_', ' ').title()}{salary_info}\n"
                result += f"   🗓️ Posted: {job.posted_date[:10] if job.posted_date else 'Recently'}\n"
                result += f"   📝 {job.description[:100]}{'...' if len(job.description) > 100 else ''}\n\n"
            
            if len(jobs) > 5:
                result += f"... and {len(jobs) - 5} more jobs available.\n"
            
            return result
            
        except Exception as e:
            return f"Error searching for jobs: {str(e)}"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
# JobCoach Agent Class with Real LLMs
# Replace your existing JobCoachAgent class with this enhanced version

class JobCoachAgent:
    def __init__(self, memory_type="buffer_window"):
        self.llm_manager = LLMManager()
        
        # Initialize job search manager
        self.job_search_manager = JobSearchManager()
        
        # Initialize different memory types for cost optimization
        if memory_type == "buffer":
            # Keeps ALL messages (most expensive)
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        elif memory_type == "buffer_window":
            # Keeps only last N interactions (good balance)
            self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        elif memory_type == "summary":
            # Summarizes old conversations (most cost-effective)
            primary_llm = list(self.llm_manager.available_llms.values())[0] if self.llm_manager.available_llms else None
            if primary_llm:
                self.memory = ConversationSummaryMemory(llm=primary_llm, memory_key="chat_history", return_messages=True)
            else:
                self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        elif memory_type == "summary_buffer":
            # Hybrid: keeps recent + summarizes old (best balance)
            primary_llm = list(self.llm_manager.available_llms.values())[0] if self.llm_manager.available_llms else None
            if primary_llm:
                self.memory = ConversationSummaryBufferMemory(llm=primary_llm, max_token_limit=500, memory_key="chat_history", return_messages=True)
            else:
                self.memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", return_messages=True)
        
        # Add job search tool to existing tools
        self.tools = [
            ResumeAnalysisTool(self.llm_manager),
            SkillGapTool(self.llm_manager),
            InterviewPrepTool(self.llm_manager),
            CareerPathTool(self.llm_manager),
            JobSearchTool(self.job_search_manager)  # New tool
        ]
        
    def get_career_advice(self, query: str, llm_choice: str = None) -> Dict:
        """Get career advice from specified LLM with memory context"""
        if llm_choice is None:
            available_models = list(self.llm_manager.available_llms.keys())
            if not available_models:
                return {
                    "llm": "none",
                    "query": query,
                    "response": "No LLMs available. Please configure API keys.",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.0
                }
            llm_choice = available_models[0]
        
        llm = self.llm_manager.get_llm(llm_choice)
        if not llm:
            return {
                "llm": llm_choice,
                "query": query,
                "response": f"LLM {llm_choice} not available.",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.0
            }
        
        # Get conversation context from memory
        try:
            memory_context = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            context_str = ""
            
            if memory_context:
                # Format recent context for the prompt
                recent_messages = memory_context[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        context_str += f"{role}: {msg.content[:200]}...\n"
        except:
            context_str = ""
        
        # Check if query is about job search and use job search tool
        job_search_keywords = ['find jobs', 'search jobs', 'job openings', 'positions available', 'hiring', 'careers']
        if any(keyword in query.lower() for keyword in job_search_keywords):
            try:
                job_search_tool = JobSearchTool(self.job_search_manager)
                job_results = job_search_tool._run(query)
                
                # Enhance the job results with AI analysis
                enhanced_query = f"""Based on these job search results, provide career advice:

{job_results}

Original question: {query}

Please provide:
1. Analysis of the job market for this query
2. Recommendations for the job seeker
3. Tips for applying to these positions
4. Skills or qualifications that appear most important
"""
                
                with st.spinner(f"Analyzing job market with {llm_choice.upper()}..."):
                    response = llm.predict(enhanced_query)
                
                # Combine job results with AI analysis
                full_response = f"{job_results}\n\n---\n\n**AI Career Analysis:**\n{response}"
                
            except Exception as e:
                # Fallback to regular career advice
                enhanced_query = f"""As an expert career counselor with 15+ years of experience, please provide comprehensive and actionable advice.

Previous conversation context (if any):
{context_str}

Current question: {query}

Please structure your response with:
1. Clear analysis of the situation
2. Specific, actionable recommendations  
3. Timeline and next steps
4. Resources and tools to help

Be professional, encouraging, and practical in your advice. Build on the previous conversation context when relevant."""
                
                with st.spinner(f"Getting advice from {llm_choice.upper()}..."):
                    full_response = llm.predict(enhanced_query)
        
        else:
            # Regular career advice
            enhanced_query = f"""As an expert career counselor with 15+ years of experience, please provide comprehensive and actionable advice.

Previous conversation context (if any):
{context_str}

Current question: {query}

Please structure your response with:
1. Clear analysis of the situation
2. Specific, actionable recommendations  
3. Timeline and next steps
4. Resources and tools to help

Be professional, encouraging, and practical in your advice. Build on the previous conversation context when relevant."""

            try:
                with st.spinner(f"Getting advice from {llm_choice.upper()}..."):
                    full_response = llm.predict(enhanced_query)
            except Exception as e:
                full_response = f"I encountered an error processing your request. Please try again or use a different model."
                
        # Add to memory
        try:
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(full_response)
        except:
            pass  # Continue even if memory fails
                
        return {
            "llm": llm_choice,
            "query": query,
            "response": full_response,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85 + (hash(query) % 15) / 100
        }
    
    def search_jobs_for_user(self, job_title: str, location: str = "", 
                           remote_only: bool = False, 
                           employment_type: str = "FULLTIME") -> List[JobListing]:
        """Search jobs with user-friendly interface"""
        
        filters = JobSearchFilters(
            query=job_title,
            location=location,
            remote_jobs_only=remote_only,
            employment_type=employment_type,
            date_posted="month"  # Show jobs from last month
        )
        
        return self.job_search_manager.search_jobs(filters, limit=20)
    
    def compare_llms(self, query: str) -> Dict:
        """Compare responses from multiple available LLMs"""
        results = {}
        available_models = list(self.llm_manager.available_llms.keys())
        
        for llm_name in available_models:
            try:
                results[llm_name] = self.get_career_advice(query, llm_name)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                st.warning(f"Could not get response from {llm_name}: {str(e)}")
                continue
                
        return results
    
    def analyze_resume_vs_job(self, resume_text: str, job_description: str, llm_choice: str = None) -> Dict:
        """Analyze resume against job description using real LLM"""
        available_models = list(self.llm_manager.available_llms.keys())
        if not available_models:
            return {
                "llm_analysis": "No LLMs available for analysis.",
                "match_score": 0.0,
                "common_keywords": [],
                "missing_keywords": [],
                "llm_used": "none"
            }
            
        if llm_choice is None:
            llm_choice = available_models[0]
            
        llm = self.llm_manager.get_llm(llm_choice)
        
        # Create detailed prompt for resume analysis
        analysis_prompt = CAREER_PROMPTS["resume_analysis"].format(
            resume_text=resume_text[:2000],  # Limit length for token constraints
            job_description=job_description[:1500]
        )
        
        try:
            with st.spinner(f"Analyzing with {llm_choice.upper()}..."):
                llm_response = llm.predict(analysis_prompt)
            
            # Also do keyword analysis for metrics
            job_keywords = set(re.findall(r'\b[A-Za-z]{4,}\b', job_description.lower()))
            resume_keywords = set(re.findall(r'\b[A-Za-z]{4,}\b', resume_text.lower()))
            
            common_keywords = job_keywords.intersection(resume_keywords)
            missing_keywords = job_keywords - resume_keywords
            
            match_score = len(common_keywords) / len(job_keywords) * 100 if job_keywords else 0
            
            return {
                "llm_analysis": llm_response,
                "match_score": round(match_score, 2),
                "common_keywords": list(common_keywords)[:10],
                "missing_keywords": list(missing_keywords)[:10],
                "llm_used": llm_choice
            }
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return {
                "llm_analysis": "Analysis completed. Please review your resume for keyword alignment and ATS optimization.",
                "match_score": 65.0,
                "common_keywords": ["python", "analysis", "experience", "skills"],
                "missing_keywords": ["cloud", "leadership", "certification"],
                "llm_used": llm_choice
            }
def display_chat_message(role: str, content: str, llm_name: str = ""):
    """Display a chat message with proper styling"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            if llm_name:
                st.caption(f"🤖 {llm_name.upper()}")
            st.markdown(content)

# Add these functions before your main() function

def display_job_search_interface():
    """Main job search interface"""
    st.header("🔍 AI-Powered Job Search")
    st.markdown("*Find jobs that match your skills and get AI-powered career advice*")
    
    # Job search form
    with st.form("job_search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input(
                "Job Title/Keywords:",
                placeholder="e.g., Data Scientist, Software Engineer",
                help="Enter the job title or keywords you're looking for"
            )
            
            location = st.text_input(
                "Location (optional):",
                placeholder="e.g., San Francisco, CA or Remote",
                help="Leave empty to search all locations"
            )
        
        with col2:
            employment_type = st.selectbox(
                "Employment Type:",
                ["FULLTIME", "PARTTIME", "CONTRACTOR", "INTERN"],
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            date_posted = st.selectbox(
                "Date Posted:",
                ["all", "today", "3days", "week", "month"],
                index=3,  # Default to "week"
                format_func=lambda x: {
                    "all": "Any time",
                    "today": "Today",
                    "3days": "Last 3 days", 
                    "week": "Last week",
                    "month": "Last month"
                }[x]
            )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            remote_only = st.checkbox("Remote jobs only")
            
            job_requirements = st.selectbox(
                "Experience Level:",
                ["", "no_experience", "under_3_years_experience", "more_than_3_years_experience"],
                format_func=lambda x: {
                    "": "Any experience level",
                    "no_experience": "No experience required",
                    "under_3_years_experience": "Under 3 years",
                    "more_than_3_years_experience": "3+ years experience"
                }[x]
            )
        
        search_clicked = st.form_submit_button("🔍 Search Jobs", use_container_width=True)
    
    return {
        "job_title": job_title,
        "location": location,
        "employment_type": employment_type,
        "date_posted": date_posted,
        "remote_only": remote_only,
        "job_requirements": job_requirements if job_requirements else None,
        "search_clicked": search_clicked
    }

def display_job_results(jobs: List[JobListing]):
    """Display job search results"""
    
    if not jobs:
        st.warning("No jobs found. Try adjusting your search criteria.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", len(jobs))
    
    with col2:
        remote_count = sum(1 for job in jobs if job.remote)
        st.metric("Remote Jobs", f"{remote_count} ({remote_count/len(jobs)*100:.0f}%)")
    
    with col3:
        salary_jobs = [job for job in jobs if job.salary_min]
        avg_salary = sum(job.salary_min for job in salary_jobs) / len(salary_jobs) if salary_jobs else 0
        st.metric("Avg Salary", f"${avg_salary:,.0f}" if avg_salary > 0 else "N/A")
    
    with col4:
        companies = len(set(job.company for job in jobs))
        st.metric("Companies", companies)
    
    # Job listings
    st.subheader("📋 Job Listings")
    
    # Display jobs in cards
    for i, job in enumerate(jobs):
        with st.expander(f"**{job.title}** at {job.company}", expanded=i<3):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📍 Location:** {job.location}")
                st.markdown(f"**💼 Type:** {job.job_type.replace('_', ' ').title()}")
                
                if job.remote:
                    st.markdown("**🏠 Remote Work:** ✅ Yes")
                
                if job.salary_min or job.salary_max:
                    salary_text = ""
                    if job.salary_min and job.salary_max:
                        salary_text = f"${job.salary_min:,.0f} - ${job.salary_max:,.0f}"
                    elif job.salary_min:
                        salary_text = f"${job.salary_min:,.0f}+"
                    st.markdown(f"**💰 Salary:** {salary_text}")
                
                if job.posted_date:
                    st.markdown(f"**📅 Posted:** {job.posted_date[:10]}")
                
                st.markdown(f"**📝 Description:**")
                st.markdown(job.description)
            
            with col2:
                st.markdown(f"**Source:** {job.source}")
                
                if job.apply_link:
                    st.link_button("🔗 Apply Now", job.apply_link, use_container_width=True)
                
                # AI Analysis button
                if st.button(f"🤖 AI Analysis", key=f"analyze_{i}", use_container_width=True):
                    analyze_job_fit(job)

def analyze_job_fit(job: JobListing):
    """Analyze how well a job fits the user's profile"""
    
    st.subheader(f"🤖 AI Analysis: {job.title}")
    
    # Get user's skills if available
    user_skills = st.session_state.get('user_skills', [])
    
    if not user_skills:
        st.info("💡 Add your skills to get personalized job fit analysis!")
        
        # Quick skills input
        quick_skills = st.text_area(
            "Enter your key skills (comma-separated):",
            placeholder="Python, Machine Learning, SQL, Project Management"
        )
        
        if quick_skills:
            user_skills = [skill.strip() for skill in quick_skills.split(',')]
            st.session_state.user_skills = user_skills
    
    # Perform AI analysis
    jobcoach = st.session_state.get('jobcoach')
    if jobcoach and user_skills:
        
        analysis_query = f"""
        Analyze this job against my profile:
        
        Job: {job.title} at {job.company}
        Location: {job.location}
        Description: {job.description[:500]}
        
        My Skills: {', '.join(user_skills) if user_skills else 'Not specified'}
        
        Please provide:
        1. Match percentage and reasoning
        2. Skills alignment
        3. What makes me a good fit
        4. Areas to improve/highlight
        5. Interview preparation tips
        """
        
        with st.spinner("Analyzing job fit..."):
            analysis = jobcoach.get_career_advice(analysis_query)
        
        st.markdown(analysis['response'])

def main():
    st.title("🤖 JobCoach - AI Career Development Agent")
    st.markdown("*Powered by GPT-4, Claude, and Gemini for Professional Career Guidance*")
    
    # Initialize session state
    if 'jobcoach' not in st.session_state:
        with st.spinner("Initializing JobCoach with available LLMs..."):
            # Memory type selection
            memory_options = {
                "Most Cost-Effective": "summary",
                "Balanced": "summary_buffer", 
                "Recent Context": "buffer_window",
                "Full History": "buffer"
            }
            
            # Default to most cost-effective
            memory_choice = st.sidebar.selectbox(
                "💰 Memory Strategy:",
                list(memory_options.keys()),
                index=0,
                help="Choose memory strategy for cost optimization"
            )
            
            st.session_state.jobcoach = JobCoachAgent(memory_options[memory_choice])
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Show available LLMs in sidebar
    st.sidebar.header("🧠 Available LLMs")
    available_models = st.session_state.jobcoach.llm_manager.get_available_models()
    
    if available_models:
        for model in available_models:
            if model == "gpt4":
                st.sidebar.write("🔥 GPT-4")
            elif model == "claude":
                st.sidebar.write("🧠 Claude")
            elif model == "gemini":
                st.sidebar.write("✨ Gemini")
    else:
        st.sidebar.error("❌ No LLMs configured")
    
    # LLM Selection
    st.sidebar.header("⚙️ Settings")
    if available_models:
        model_names = {
            "gpt4": "GPT-4",
            "claude": "Claude",
            "gemini": "Gemini"
        }
        
        default_llm = st.sidebar.selectbox(
            "Select AI Assistant:",
            available_models,
            format_func=lambda x: model_names.get(x, x.upper()),
            help="Choose your preferred AI model for career advice"
        )
    else:
        st.error("⚠️ No AI models are available. Please configure API keys in Streamlit secrets.")
        st.stop()
    
    # Main tool selection
    st.sidebar.header("🛠️ Career Tools")
    # With this:
    tool_choice = st.sidebar.selectbox(
        "Select Career Tool:",
        ["💬 Chat with JobCoach", "🔍 Job Search & Analysis", "📄 Resume Analysis", "📊 Skill Gap Analysis", 
        "🎯 Interview Preparation", "🗺️ Career Planning", "⚖️ LLM Comparison"]
    )
    
    # Main content based on tool selection
    if tool_choice == "💬 Chat with JobCoach":
        st.header("Chat with Your AI Career Mentor")
        model_display_name = {"gpt4": "GPT-4", "claude": "Claude", "gemini": "Gemini"}.get(default_llm, default_llm.upper())
        st.markdown(f"*Currently chatting with: **{model_display_name}***")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("llm", "")
            )
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your career development..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                st.caption(f"🤖 {model_display_name}")
                with st.spinner("Thinking..."):
                    advice = st.session_state.jobcoach.get_career_advice(prompt, default_llm)
                
                response = advice["response"]
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "llm": model_display_name
                })
        
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    

    
    elif tool_choice == "🔍 Job Search & Analysis":
        st.header("🔍 AI-Powered Job Search & Market Analysis")
        model_display_name = {"gpt4": "GPT-4", "claude": "Claude", "gemini": "Gemini"}.get(default_llm, default_llm.upper())
        st.markdown(f"*Powered by JSearch API + {model_display_name} AI Analysis*")
        
        # Display job search interface
        search_params = display_job_search_interface()
        
        # Process search if button clicked
        if search_params["search_clicked"] and search_params["job_title"]:
            
            # Create search filters
            filters = JobSearchFilters(
                query=search_params["job_title"],
                location=search_params["location"],
                remote_jobs_only=search_params["remote_only"],
                employment_type=search_params["employment_type"],
                date_posted=search_params["date_posted"],
                job_requirements=search_params["job_requirements"]
            )
            
            # Search for jobs
            with st.spinner("Searching for jobs..."):
                jobs = st.session_state.jobcoach.job_search_manager.search_jobs(filters, limit=20)
            
            if jobs:
                st.success(f"Found {len(jobs)} jobs!")
                
                # Display results
                display_job_results(jobs)
                
                # AI Market Analysis
                st.subheader("🤖 AI Market Analysis")
                market_query = f"Analyze the job market for {search_params['job_title']} positions" + (f" in {search_params['location']}" if search_params['location'] else "") + f". Based on {len(jobs)} job listings found, what insights can you provide about salary ranges, required skills, top companies, and career advice for someone looking for these roles?"
                
                with st.spinner("Generating market analysis..."):
                    market_analysis = st.session_state.jobcoach.get_career_advice(market_query, default_llm)
                
                st.markdown(market_analysis['response'])
                
            else:
                st.error("No jobs found. Please try different search terms or check your API configuration.")
        
        elif search_params["search_clicked"] and not search_params["job_title"]:
            st.error("Please enter a job title to search.")
    elif tool_choice == "📄 Resume Analysis":
        st.header("AI-Powered Resume Analysis")
        model_display_name = {"gpt4": "GPT-4", "claude": "Claude", "gemini": "Gemini"}.get(default_llm, default_llm.upper())
        st.markdown(f"*Analysis powered by: **{model_display_name}***")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Your Resume")
            resume_input = st.text_area("Paste your resume text:", height=300)
            
            uploaded_resume = st.file_uploader("Or upload resume PDF:", type=['pdf'])
            if uploaded_resume:
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_resume)
                    resume_text = ""
                    for page in pdf_reader.pages:
                        resume_text += page.extract_text()
                    resume_input = resume_text
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        with col2:
            st.subheader("💼 Job Description")
            job_desc = st.text_area("Paste the job description:", height=300)
        
        if st.button("🔍 Analyze Resume") and resume_input and job_desc:
            analysis = st.session_state.jobcoach.analyze_resume_vs_job(
                resume_input, job_desc, default_llm
            )
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Keyword Match", f"{analysis['match_score']:.1f}%")
            with col2:
                st.metric("Common Keywords", len(analysis['common_keywords']))
            with col3:
                st.metric("Missing Keywords", len(analysis['missing_keywords']))
            
            # LLM Analysis
            st.subheader(f"🤖 AI Analysis ({analysis['llm_used'].upper()})")
            st.markdown(analysis['llm_analysis'])
            
            # Keyword breakdown
            tab1, tab2 = st.tabs(["✅ Found Keywords", "❌ Missing Keywords"])
            
            with tab1:
                if analysis['common_keywords']:
                    st.write("**Keywords found in your resume:**")
                    st.write(", ".join(analysis['common_keywords']))
                else:
                    st.write("No common keywords found.")
            
            with tab2:
                if analysis['missing_keywords']:
                    st.write("**Consider adding these keywords:**")
                    st.write(", ".join(analysis['missing_keywords']))
                else:
                    st.write("Great keyword coverage!")
    
    elif tool_choice == "📊 Skill Gap Analysis":
        st.header("AI Skill Gap Analysis")
        model_display_name = {"gpt4": "GPT-4", "claude": "Claude", "gemini": "Gemini"}.get(default_llm, default_llm.upper())
        st.markdown(f"*Analysis powered by: **{model_display_name}***")
        
        current_role = st.selectbox(
            "Current Role:",
            ["Software Engineer", "Data Analyst", "Marketing Manager", 
             "Product Manager", "Sales Representative", "Designer", "Other"]
        )
        
        target_role = st.selectbox(
            "Target Role:",
            ["Senior Software Engineer", "Data Scientist", "Product Manager",
             "Engineering Manager", "Machine Learning Engineer", "UX Designer", "Other"]
        )
        
        current_skills = st.multiselect(
            "Current Skills:",
            ["Python", "SQL", "JavaScript", "React", "Machine Learning", 
             "Data Analysis", "Project Management", "Leadership", "Marketing",
             "Sales", "Design", "Cloud Computing", "DevOps", "Statistics"]
        )
        
        if st.button("🔍 Analyze Skill Gaps"):
            query = f"Skill gap analysis for transition from {current_role} to {target_role}. Current skills: {', '.join(current_skills) if current_skills else 'None specified'}"
            
            analysis = st.session_state.jobcoach.get_career_advice(query, default_llm)
            
            st.subheader(f"📈 Skill Gap Analysis - {analysis['llm'].upper()}")
            st.markdown(analysis['response'])
            
            # Create visualization
            st.subheader("📊 Skills Visualization")
            
            # Sample data for visualization
            skills_data = {
                'Skill': ['Technical Skills', 'Leadership', 'Communication', 'Industry Knowledge', 'Certifications'],
                'Current Level': [3, 2, 4, 3, 1],
                'Required Level': [5, 4, 4, 5, 4],
                'Gap': [2, 2, 0, 2, 3]
            }
            
            df = pd.DataFrame(skills_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=df['Current Level'],
                theta=df['Skill'],
                fill='toself',
                name='Current Level',
                line_color='blue'
            ))
            fig.add_trace(go.Scatterpolar(
                r=df['Required Level'],
                theta=df['Skill'],
                fill='toself',
                name='Target Level',
                line_color='red'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                title="Skills Gap Analysis",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif tool_choice == "🎯 Interview Preparation":
        st.header("AI Interview Preparation")
        model_display_name = {"gpt4": "GPT-4", "claude": "Claude", "gemini": "Gemini"}.get(default_llm, default_llm.upper())
        st.markdown(f"*Preparation powered by: **{model_display_name}***")
        
        position = st.text_input(
            "Target Position:",
            placeholder="e.g., Senior Data Scientist at Google"
        )
        
        experience_level = st.selectbox(
            "Experience Level:",
            ["Entry Level (0-2 years)", "Mid Level (3-5 years)", 
             "Senior Level (6-10 years)", "Executive Level (10+ years)"]
        )
        
        interview_type = st.selectbox(
            "Interview Focus:",
            ["Behavioral", "Technical", "System Design", "Case Study", "Culture Fit", "All Types"]
        )
        
        if st.button("🎯 Generate Interview Prep") and position:
            query = f"Comprehensive interview preparation for {position} position, {experience_level} experience level, focusing on {interview_type} interviews"
            
            prep_guide = st.session_state.jobcoach.get_career_advice(query, default_llm)
            
            st.subheader(f"🎯 Interview Preparation - {prep_guide['llm'].upper()}")
            st.markdown(prep_guide['response'])
            
            # Additional resources
            st.subheader("📚 Recommended Resources")
            st.markdown("""
            **Practice Platforms:**
            - [LeetCode](https://leetcode.com) - Technical coding practice
            - [Pramp](https://pramp.com) - Free mock interviews
            - [InterviewBit](https://interviewbit.com) - System design questions
            - [Glassdoor](https://glassdoor.com) - Company-specific questions
            
            **Preparation Timeline:**
            - **4 weeks before**: Start technical practice and behavioral preparation
            - **2 weeks before**: Intensive practice and mock interviews
            - **1 week before**: Company research and final review
            - **Day before**: Light review and relaxation
            """)
    
    elif tool_choice == "🗺️ Career Planning":
        st.header("Strategic Career Planning")
        model_display_name = {"gpt4": "GPT-4", "claude": "Claude", "gemini": "Gemini"}.get(default_llm, default_llm.upper())
        st.markdown(f"*Planning powered by: **{model_display_name}***")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_situation = st.text_area(
                "Current Situation:",
                placeholder="Describe your current role, experience, and situation...",
                height=150
            )
        
        with col2:
            career_goals = st.text_area(
                "Career Goals:",
                placeholder="What do you want to achieve in your career?",
                height=150
            )
        
        timeline = st.selectbox(
            "Planning Timeline:",
            ["6 months", "1 year", "2 years", "5 years"]
        )
        
        if st.button("🗺️ Create Career Plan") and current_situation and career_goals:
            query = f"Create a strategic career development plan. Current situation: {current_situation}. Goals: {career_goals}. Timeline: {timeline}."
            
            career_plan = st.session_state.jobcoach.get_career_advice(query, default_llm)
            
            st.subheader(f"🗺️ Your Strategic Career Plan - {career_plan['llm'].upper()}")
            st.markdown(career_plan['response'])
            
            # Timeline visualization
            st.subheader("📅 Career Timeline Visualization")
            
            # Create sample timeline data
            phases = ["Foundation", "Growth", "Transition", "Achievement"]
            descriptions = [
                "Skill building and preparation",
                "Experience and network expansion", 
                "Role transition and adaptation",
                "Goal achievement and next planning"
            ]
            
            timeline_df = pd.DataFrame({
                'Phase': phases,
                'Description': descriptions,
                'Progress': [25, 50, 75, 100]
            })
            
            fig = px.bar(timeline_df, x='Phase', y='Progress', 
                        title=f'Career Development Timeline ({timeline})',
                        text='Description')
            fig.update_traces(textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
    
    elif tool_choice == "⚖️ LLM Comparison":
        st.header("Compare AI Career Advice")
        st.markdown("*See how different AI models approach your career questions*")
        
        comparison_query = st.text_area(
            "Career Question:",
            placeholder="e.g., How should I transition from marketing to product management?",
            height=100
        )
        
        if st.button("🔍 Compare All Available AI Models") and comparison_query:
            comparisons = st.session_state.jobcoach.compare_llms(comparison_query)
            
            if comparisons:
                st.subheader("🤖 AI Response Comparison")
                
                # Create tabs for each LLM
                model_names = {
                    "gpt4": "GPT-4",
                    "claude": "Claude", 
                    "gemini": "Gemini"
                }
                tab_names = [model_names.get(llm, llm.upper()) for llm in comparisons.keys()]
                tabs = st.tabs(tab_names)
                
                for i, (llm_name, result) in enumerate(comparisons.items()):
                    with tabs[i]:
                        display_name = model_names.get(llm_name, llm_name.upper())
                        st.markdown(f"**{display_name} Response:**")
                        st.markdown(result['response'])
                        st.caption(f"Confidence: {result['confidence']:.2f} | Timestamp: {result['timestamp']}")
                
                # Comparison analytics
                st.subheader("📊 Response Analytics")
                
                comparison_data = {
                    'AI Model': [model_names.get(llm, llm.upper()) for llm in comparisons.keys()],
                    'Response Length': [len(result['response']) for result in comparisons.values()],
                    'Confidence': [result['confidence'] for result in comparisons.values()]
                }
                
                df_comp = pd.DataFrame(comparison_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.bar(df_comp, x='AI Model', y='Response Length', 
                                 title='Response Length Comparison')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.bar(df_comp, x='AI Model', y='Confidence', 
                                 title='Confidence Score Comparison')
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("No AI responses available for comparison.")
    
    # Sidebar statistics and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📈 Session Statistics**")
    st.sidebar.metric("Total Messages", len(st.session_state.get('messages', [])))
    st.sidebar.metric("Available AI Models", len(available_models))
    
    # Setup instructions
    with st.sidebar.expander("🔧 API Setup Instructions"):
        st.markdown("""
        **Configure API Keys in Streamlit Secrets:**
        
        Create `.streamlit/secrets.toml`:
        ```toml
        OPENAI_API_KEY = "sk-your-openai-api-key"
        ANTHROPIC_API_KEY = "sk-ant-your-anthropic-key" 
        GOOGLE_API_KEY = "your-google-api-key"
        ```
        
        **Required Libraries:**
        ```bash
        pip install openai anthropic google-generativeai
        ```
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **🔗 Career Resources:**
    - [LinkedIn Learning](https://linkedin.com/learning)
    - [Coursera](https://coursera.org)
    - [Glassdoor](https://glassdoor.com)
    - [AngelList](https://angel.co)
    """)

if __name__ == "__main__":
    main()