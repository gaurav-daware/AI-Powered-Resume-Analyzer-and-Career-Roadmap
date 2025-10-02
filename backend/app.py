from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import spacy
import re
from PyPDF2 import PdfReader
import os
import uuid # For unique filenames
from flask_cors import CORS
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- Load environment variables for API key ---
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Initialize with Google's Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Use an appropriate error handling for production
    print("FATAL: GEMINI_API_KEY not found. Please set it in your .env file.")
    # Fallback to a stub or raise an error
genai.configure(api_key=GEMINI_API_KEY)
# Using a slightly more capable model for complex analysis tasks
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

# Initialize spacy for simple text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HIGHLIGHT_FOLDER'] = './highlighted_resumes'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(app.config['HIGHLIGHT_FOLDER']):
    os.makedirs(app.config['HIGHLIGHT_FOLDER'])

# --- Global State ---
current_resume = None       # (unique_filename, raw_text)
cleaned_resume_text = None  # Cleaned resume text for analysis
detected_domain = "General Career Field" # Dynamic domain detection result

# --- LangChain Globals ---
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
conversation_chain = None 

# --- NEW: LangChain Initialization ---
def initialize_conversation_chain(resume_text, domain):
    global conversation_chain, memory
    
    # Clear memory for the new resume
    memory.clear()

    # 1. Define the LLM (LangChain wrapper)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY,
        temperature=0.5
    )
    
    # 2. Define the Prompt Template (Includes safety and context)
    template = f"""
    You are an expert, professional career advisor specializing in the **{{domain}}** field.
    Your analysis and advice are strictly based on the following resume:
    ---
    RESUME CONTEXT:
    {{resume_text}}
    ---
    
    You must maintain a professional and objective tone.

    RULES:
    1. If the user's query is **inappropriate, offensive, or clearly non-professional and unrelated to job searching, career development, or skills**, respond ONLY with the standard, polite refusal: 'I am here to assist you with career and professional development questions only. Please submit a query related to your career goals or resume.'
    2. Use the "chat_history" to remember context from the previous 5 turns.
    3. When discussing salary, acknowledge that providing an exact figure is impossible, and focus on the **factors** that will influence their potential salary range in the detected career field.
    4. Provide detailed, actionable advice relevant to the {{domain}} field.

    {{chat_history}}
    Human: {{input}}
    AI:"""

    PROMPT = PromptTemplate(
        input_variables=["chat_history", "input"],
        # Partial variables inject the static resume/domain context into every prompt
        partial_variables={"resume_text": resume_text[:4000], "domain": domain}, 
        template=template,
    )

    # 3. Create the Conversation Chain
    conversation_chain = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        verbose=False, # Set to True for debugging the prompt structure
        memory=memory,
    )

# --- NEW: Dynamic Domain Detection (Replacing static list) ---
def detect_resume_domain(text):
    """Dynamically detect the specific career field of the resume using AI."""
    try:
        prompt = f"""
        Analyze the following resume text and identify the most specific and relevant career domain, industry, or job function. 
        Examples of possible responses are: 'Quantitative Finance', 'Biomedical Research', 'Civil Engineering', 'UX/UI Design', 'Primary School Education'.
        
        Resume Text: {text[:1500]}...
        
        Respond with **ONLY** the domain name. Do not add any explanation or quotation marks.
        """
        
        response = gemini_model.generate_content(prompt)
        domain = response.text.strip()
        
        # Simple cleanup to handle stray characters and normalize
        domain = re.sub(r'[^a-zA-Z0-9\s/&-]', '', domain) 
        if not domain:
            return "General Career Field"
            
        return domain.strip()
            
    except Exception as e:
        print(f"Error detecting domain: {e}")
        return "General Career Field"

# --- NEW: Dynamic Skill Extraction for Analysis ---
def extract_required_skills(job_requirement):
    """Use AI to extract a structured list of hard and soft skills from job requirements."""
    try:
        prompt = f"""
        Analyze the following job requirement and extract a list of 10-15 most important **hard skills** (e.g., Python, SQL, Autocad, GAAP) and **soft skills** (e.g., Leadership, Communication, Problem-Solving).

        Job Requirement: {job_requirement}

        Format your response as a single, valid JSON object with two keys: "hard_skills" and "soft_skills". Each value must be a JSON array of strings.
        Example: 
        {{
            "hard_skills": ["SQL", "Tableau", "PMP"],
            "soft_skills": ["Negotiation", "Teamwork"]
        }}
        """
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error extracting required skills: {e}")
        return {"hard_skills": [], "soft_skills": []}

def extract_resume_skills(resume_text, domain):
    """Use AI to extract skills from the resume, guided by the detected domain."""
    try:
        prompt = f"""
        Analyze the following resume in the context of the '{domain}' field. Extract a list of all technical, software, and soft skills demonstrated in the resume (limit 30 skills).

        Resume Text: {resume_text[:1500]}

        Respond with a comma-separated list of skills **ONLY**. Do not add any other text, numbers, or bullet points.
        """
        response = gemini_model.generate_content(prompt)
        
        skills_list = [s.strip() for s in response.text.split(',') if s.strip()]
        return skills_list
        
    except Exception as e:
        print(f"Error extracting resume skills: {e}")
        return []

# --- Utility Functions (Modified) ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_single_resume(text):
    if not text.strip():
        return None
    cleaned_text = re.sub(r'[\uf000-\uf8ff]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def delete_all_files():
    """Deletes all files in the upload and highlighted folders."""
    # ... (existing file deletion logic)
    for folder in [UPLOAD_FOLDER, app.config['HIGHLIGHT_FOLDER']]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

def get_ats_status(score):
    """Get ATS compatibility status based on score."""
    if score >= 85:
        return {"level": "high", "label": "Excellent Match", "color": "green"}
    elif score >= 70:
        return {"level": "medium", "label": "Strong Match", "color": "yellow"}
    elif score >= 50:
        return {"level": "medium", "label": "Good Match", "color": "orange"}
    else:
        return {"level": "low", "label": "Needs Significant Improvement", "color": "red"}

def summarize_text(text, num_sentences=3):
    """Creates a brief summary of the resume text."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return " ".join(sentences[:num_sentences])

# --- Core Dynamic Analysis Functions (Major Overhaul) ---

# Replaced the original perform_detailed_analysis to use dynamic skill extraction
def perform_detailed_analysis(resume_text, job_requirement, domain):
    """Perform comprehensive resume analysis with dynamic skill scoring."""
    
    # 1. Dynamically extract required skills from the Job Description (AI Call 1)
    required_skills_dict = extract_required_skills(job_requirement)
    required_skills = required_skills_dict.get('hard_skills', []) + required_skills_dict.get('soft_skills', [])
    
    # 2. Dynamically extract skills from the Resume (AI Call 2)
    resume_skills_list = extract_resume_skills(resume_text, domain)

    # Convert to sets for efficient comparison, normalized to lowercase
    job_keywords_set = set([k.lower().strip() for k in required_skills])
    resume_skills_set = set([s.lower().strip() for s in resume_skills_list])
    
    # 3. Calculate Matches and Missing
    matching_keywords = list(job_keywords_set.intersection(resume_skills_set))
    missing_keywords = list(job_keywords_set - resume_skills_set)

    # 4. TF-IDF for Conceptual Similarity
    documents = [resume_text, job_requirement]
    
    # Use the full text for conceptual TF-IDF (robust against varying skill names)
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    if tfidf_matrix.shape[0] < 2:
        base_score = 0
    else:
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        base_score = cosine_sim[0][0] * 100
    
    # 5. Calculate Score based on Skill Match and Conceptual Match
    skill_match_ratio = len(matching_keywords) / max(1, len(job_keywords_set))
    
    # Weighted score: 70% skill match (direct alignment), 30% conceptual match (overall fit)
    final_score = (base_score * 0.30) + (skill_match_ratio * 70)
    final_score = min(100, final_score) 
    
    # 6. Calculate Keyword Density
    keyword_density = {}
    for keyword in matching_keywords:
        count = resume_text.lower().count(keyword)
        keyword_density[keyword] = count
    
    # 7. Generate AI-powered insights (Now fully dynamic, removing old domain-specific helpers)
    recommendations = generate_recommendations(resume_text, job_requirement, final_score, missing_keywords, domain)
    skill_gaps = generate_skill_gap_analysis(resume_text, job_requirement, domain)
    
    return {
        "score": round(final_score, 2),
        "summary": summarize_text(resume_text),
        "ats_status": get_ats_status(final_score),
        "keyword_analysis": {
            "matching_keywords": matching_keywords[:15],
            "missing_keywords": missing_keywords[:10],
            "keyword_density": dict(sorted(keyword_density.items(), key=lambda x: x[1], reverse=True)[:8])
        },
        "recommendations": recommendations,
        "skill_gaps": skill_gaps
    }

def generate_recommendations(resume_text, job_requirement, score, missing_keywords, domain):
    """Generate AI-powered dynamic recommendations."""
    try:
        prompt = f"""
        As an expert resume consultant specializing in the **{domain}** field, analyze this resume against the job requirements and provide specific, actionable recommendations to improve its ATS score and relevance.

        First, provide a brief (one sentence) **Domain-Specific Guidance** for a resume in **{domain}**. 
        
        Resume Score: {score:.1f}%
        Missing Keywords (Top 5): {', '.join(missing_keywords[:5])}

        Resume Text: {resume_text[:1500]}...
        Job Requirements: {job_requirement[:1000]}...

        Provide exactly 4 specific recommendations in this format:
        1. [Category]: [Specific actionable advice]

        Categories should be relevant to **{domain}**, such as: Keywords, Quantification, Action Verbs, Formatting, Structure, Skills, Experience, Certifications, Projects.
        Keep each recommendation under 120 characters and prioritize the most impactful changes.
        """
        
        response = gemini_model.generate_content(prompt)
        recommendations_text = response.text.strip()
        
        # Parse the recommendations, looking for numbered lists
        recommendations = [line.strip() for line in recommendations_text.split('\n') if line.strip() and re.match(r'^\d\.', line.strip())]
        
        return recommendations[:4]
        
    except Exception as e:
        print(f"Error generating AI recommendations: {e}")
        return [
            "Keywords: Integrate missing keywords from the job description.",
            "Quantification: Add numbers and metrics to your accomplishments.",
            "Action Verbs: Start bullet points with strong action verbs.",
            "Review: Ensure all experience is relevant to the job's domain."
        ]

def generate_skill_gap_analysis(resume_text, job_requirement, domain):
    """Generate dynamic skill gap analysis with learning resources."""
    try:
        
        prompt = f"""
        As a career development expert specializing in **{domain}**, analyze the skill gaps between this resume and the job requirements.
        
        In the **Learning Resources** section for each skill gap, suggest 1-2 specific and realistic resources (courses, certifications, platforms, industry groups, etc.) **appropriate for the {domain} field**.
        
        Resume: {resume_text[:1000]}...
        Job Requirements: {job_requirement[:800]}...
        
        Identify:
        1. Current Skills: List 5-8 key skills from the resume relevant to {domain}.
        2. Skill Gaps: List 3-5 important skills missing but required for the job in {domain}.
        3. Learning Resources: Suggest resources for each gap.
        
        Format your response as a single, valid JSON object with the keys "current_skills" and "skill_gaps".
        Example:
        {{
            "current_skills": ["skill1", "skill2"],
            "skill_gaps": [
                {{
                    "skill": "skill_name",
                    "importance": "high/medium/low",
                    "resources": ["resource1 - Platform", "resource2 - Certification"]
                }}
            ]
        }}
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error generating skill gap analysis: {e}")
        return {
            "current_skills": ["Communication", "Teamwork"],
            "skill_gaps": [
                {
                    "skill": "Crucial Domain Skill",
                    "importance": "high",
                    "resources": [f"Relevant Course for {domain}", "Professional Certification for your field"]
                }
            ]
        }


# --- Flask Routes (Updated) ---

@app.route('/upload', methods=['POST'])
def upload_resume():
    global current_resume, cleaned_resume_text, detected_domain
    
    delete_all_files()

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Use a unique filename to prevent conflicts
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(file_path)
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            if not text.strip():
                return jsonify({"error": "Could not extract text from PDF. Please ensure it's a valid PDF with readable text."}), 400
            
            cleaned_resume_text = process_single_resume(text)
            current_resume = (unique_filename, text.strip())
            
            # Detect domain automatically (Dynamic AI Call)
            detected_domain = detect_resume_domain(cleaned_resume_text)
            
            # --- NEW: Initialize LangChain Chain after successful upload/detection ---
            initialize_conversation_chain(cleaned_resume_text, detected_domain)

            response = {
                "message": "Resume uploaded and processed successfully",
                "filename": original_filename,
                "parsed_text_length": len(cleaned_resume_text),
                "detected_domain": detected_domain
            }
            return jsonify(response), 201
            
        except Exception as e:
            if os.path.exists(file_path):
                 os.remove(file_path)
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

@app.route('/delete', methods=['POST'])
def delete_resume():
    global current_resume, cleaned_resume_text, detected_domain, conversation_chain
    
    if current_resume is None:
        return jsonify({"message": "No resume to delete"}), 404

    filename = current_resume[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    
    # --- NEW: Clear global state and LangChain artifacts ---
    current_resume = None
    cleaned_resume_text = None
    detected_domain = "General Career Field"
    conversation_chain = None
    
    return jsonify({"message": f"File '{filename.split('_', 1)[-1]}' deleted successfully"}), 200

@app.route('/career_roadmap', methods=['POST'])
def career_roadmap():
    global conversation_chain

    if not conversation_chain:
        return jsonify({"error": "Chat not initialized. Please upload a resume first."}), 400

    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400
    
    try:
        # --- NEW: Invoke the LangChain conversation chain ---
        response = conversation_chain.invoke(user_query)
        
        # LangChain's response object contains the text in the 'response' key
        ai_response = response.get('response', "Sorry, I couldn't process that request.")
        
        # We can extract the domain from the prompt's partial variables for the frontend
        current_domain = conversation_chain.prompt.partial_variables.get("domain", "General Career Field")

        return jsonify({
            "response": ai_response,
            "domain": current_domain
        }), 200

    except Exception as e:
        print(f"Error invoking LangChain: {e}")
        return jsonify({"error": f"An error occurred with the AI model: {str(e)}"}), 500

@app.route('/rate_resumes', methods=['POST'])
def rate_resumes():
    global cleaned_resume_text
    
    if not cleaned_resume_text:
        return jsonify({"error": "No resume available to rate"}), 400

    data = request.get_json()
    if not data or 'job_requirement' not in data:
        return jsonify({"error": "Missing 'job_requirement' in request body"}), 400

    job_requirement = data['job_requirement']
    if not job_requirement.strip():
        return jsonify({"error": "Job requirement cannot be empty"}), 400

    # The analysis functions now use the dynamic logic
    analysis_result = perform_detailed_analysis(cleaned_resume_text, job_requirement, detected_domain)
    
    response = {
        "job_requirement": job_requirement,
        "resume_count": 1,
        "detected_domain": detected_domain,
        "results": [{
            "filename": current_resume[0].split('_', 1)[-1], # Display original name
            **analysis_result
        }]
    }
    return jsonify(response), 200

if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)