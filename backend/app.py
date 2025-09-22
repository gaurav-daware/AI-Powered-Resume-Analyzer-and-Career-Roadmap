from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import spacy
import re
from PyPDF2 import PdfReader
import os
from spacy.matcher import Matcher
from flask_cors import CORS
from dotenv import load_dotenv

import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- [MODIFICATION] Load environment variables for API key ---
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- [MODIFICATION] Initialize with Google's Gemini API for the chatbot ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')

nlp = spacy.load("en_core_web_sm")

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HIGHLIGHT_FOLDER'] = './highlighted_resumes'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(app.config['HIGHLIGHT_FOLDER']):
    os.makedirs(app.config['HIGHLIGHT_FOLDER'])

current_resume = None
cleaned_resume_text = None
processed_resume_tokens = None

TECH_ENTITIES = [
    "Python", "Java", "C++", "JavaScript", "PHP", "SQL", "MySQL", "MongoDB", "SQLite", "BigQuery",
    "PostgreSQL", "HTML", "CSS", "Dart", "R", "Ruby", "Swift", "Kotlin", "TypeScript",
    "NumPy", "Pandas", "SciPy", "Dask", "GeoPandas", "Sklearn", "NLTK", "OpenCV", "Keras",
    "TensorFlow", "Pytorch", "AzureML", "Matplotlib", "Seaborn", "Plotly", "Flask", "Django",
    "Node.js", "React", "Flutter", "Bootstrap", "CodeIgniter", "REST API", "WebRTC", "MLOps",
    "Docker", "Kubernetes", "Kubeflow", "AWS", "GCP", "Azure", "Google Colab", "Heroku",
    "Jupyter", "Git", "Github", "VSCode", "Machine Learning", "Deep Learning", "NLP",
    "Computer Vision", "Data Engineering", "IoT", "Cloud Computing", "AutoCAD",
]

matcher = Matcher(nlp.vocab)
for tech in TECH_ENTITIES:
    pattern = [{"LOWER": tech.lower()}]
    matcher.add(tech, [pattern])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_single_resume(text):
    if not text.strip():
        return None, None
    
    cleaned_text = re.sub(r'[\uf000-\uf8ff]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    doc = nlp(cleaned_text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return cleaned_text, tokens

@app.route('/upload', methods=['POST'])
def upload_resume():
    global current_resume, cleaned_resume_text, processed_resume_tokens
    
    print("[Backend] Upload endpoint called")
    
    delete_all_files()

    if 'file' not in request.files:
        print("[Backend] No file part in request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    print(f"[Backend] File received: {file.filename}")
    
    if file.filename == '':
        print("[Backend] No file selected")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            print(f"[Backend] File saved to: {file_path}")

            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            print(f"[Backend] Extracted text length: {len(text)}")
            
            if not text.strip():
                print("[Backend] No text extracted from PDF")
                return jsonify({"error": "Could not extract text from PDF. Please ensure it's a valid PDF with readable text."}), 400
            
            current_resume = (filename, text.strip())
            cleaned_resume_text, processed_resume_tokens = process_single_resume(current_resume[1])

            if not cleaned_resume_text:
                print("[Backend] Failed to process resume text")
                return jsonify({"error": "Failed to process resume text"}), 400

            response = {
                "message": "Resume uploaded and processed successfully",
                "filename": filename,
                "parsed_text_length": len(cleaned_resume_text)
            }
            print(f"[Backend] Upload successful: {response}")
            return jsonify(response), 201
            
        except Exception as e:
            print(f"[Backend] Error processing file: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        print("[Backend] Invalid file type")
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

@app.route('/delete', methods=['POST'])
def delete_resume():
    global current_resume, cleaned_resume_text, processed_resume_tokens
    
    if current_resume is None:
        return jsonify({"message": "No resume to delete"}), 404

    filename = current_resume[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    
    current_resume = None
    cleaned_resume_text = None
    processed_resume_tokens = None
    
    return jsonify({"message": f"File '{filename}' deleted successfully"}), 200

def delete_all_files():
    """Deletes all files in the upload and highlighted folders."""
    for folder in [UPLOAD_FOLDER, app.config['HIGHLIGHT_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

@app.route('/career_roadmap', methods=['POST'])
def career_roadmap():
    if not cleaned_resume_text:
        return jsonify({"error": "Please upload a resume first."}), 400

    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400
    
    try:
        prompt = (
            f"You are an expert career advisor. Based on the following resume text, "
            f"provide a career roadmap and suggestions. The user is asking: '{user_query}'\n\n"
            f"Resume Text:\n{cleaned_resume_text}\n\n"
            f"Suggestions should include skill gaps, recommended courses/certifications, "
            f"and potential next career steps. Format your response clearly."
        )
        
        response = gemini_model.generate_content(prompt)
        
        return jsonify({"response": response.text}), 200

    except Exception as e:
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

    analysis_result = perform_detailed_analysis(cleaned_resume_text, job_requirement)
    
    response = {
        "job_requirement": job_requirement,
        "resume_count": 1,
        "results": [{
            "filename": current_resume[0],
            **analysis_result
        }]
    }
    return jsonify(response), 200

def perform_detailed_analysis(resume_text, job_requirement):
    """Perform comprehensive resume analysis with detailed breakdown"""
    
    # Basic TF-IDF similarity score
    documents = [resume_text, job_requirement]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    base_score = cosine_sim[0][0] * 100
    
    # Extract keywords from job requirement
    job_doc = nlp(job_requirement.lower())
    job_keywords = set()
    
    # Extract important terms (nouns, proper nouns, and tech terms)
    for token in job_doc:
        if (token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2 and 
            not token.is_stop and not token.is_punct):
            job_keywords.add(token.text)
    
    # Add tech entities from job requirement
    job_matches = matcher(job_doc)
    for match_id, start, end in job_matches:
        span = job_doc[start:end]
        job_keywords.add(span.text.lower())
    
    # Extract keywords from resume
    resume_doc = nlp(resume_text.lower())
    resume_keywords = set()
    
    for token in resume_doc:
        if (token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2 and 
            not token.is_stop and not token.is_punct):
            resume_keywords.add(token.text)
    
    # Add tech entities from resume
    resume_matches = matcher(resume_doc)
    for match_id, start, end in resume_matches:
        span = resume_doc[start:end]
        resume_keywords.add(span.text.lower())
    
    # Calculate keyword matches
    matching_keywords = list(job_keywords.intersection(resume_keywords))
    missing_keywords = list(job_keywords - resume_keywords)
    
    # Calculate keyword density for top skills
    keyword_density = {}
    for keyword in matching_keywords[:10]:  # Top 10 matching keywords
        count = resume_text.lower().count(keyword)
        keyword_density[keyword] = count
    
    # Generate AI-powered recommendations
    recommendations = generate_recommendations(resume_text, job_requirement, base_score, missing_keywords)
    
    # Generate skill gap analysis
    skill_gaps = generate_skill_gap_analysis(resume_text, job_requirement)
    
    return {
        "score": round(base_score, 2),
        "summary": summarize_text(resume_text),
        "ats_status": get_ats_status(base_score),
        "keyword_analysis": {
            "matching_keywords": matching_keywords[:15],  # Top 15 matches
            "missing_keywords": missing_keywords[:10],    # Top 10 missing
            "keyword_density": dict(sorted(keyword_density.items(), key=lambda x: x[1], reverse=True)[:8])
        },
        "recommendations": recommendations,
        "skill_gaps": skill_gaps
    }

def get_ats_status(score):
    """Get ATS compatibility status"""
    if score >= 80:
        return {"level": "high", "label": "Excellent Match", "color": "green"}
    elif score >= 60:
        return {"level": "medium", "label": "Good Match", "color": "yellow"}
    else:
        return {"level": "low", "label": "Needs Improvement", "color": "red"}

def generate_recommendations(resume_text, job_requirement, score, missing_keywords):
    """Generate AI-powered recommendations"""
    try:
        prompt = f"""
        As an expert resume consultant, analyze this resume against the job requirements and provide specific, actionable recommendations.
        
        Resume Score: {score:.1f}%
        Missing Keywords: {', '.join(missing_keywords[:5])}
        
        Resume Text: {resume_text[:1500]}...
        Job Requirements: {job_requirement[:1000]}...
        
        Provide exactly 4-6 specific recommendations in this format:
        1. [Category]: [Specific actionable advice]
        
        Categories should include: Keywords, Quantification, Action Verbs, Structure, Skills, Experience
        Keep each recommendation under 100 characters.
        """
        
        response = gemini_model.generate_content(prompt)
        recommendations_text = response.text.strip()
        
        # Parse recommendations into list
        recommendations = []
        for line in recommendations_text.split('\n'):
            if line.strip() and (line.strip().startswith(tuple('123456789')) or '.' in line[:3]):
                recommendations.append(line.strip())
        
        return recommendations[:6]  # Limit to 6 recommendations
        
    except Exception as e:
        # Fallback recommendations based on score
        if score < 60:
            return [
                "Keywords: Add more relevant keywords from the job description",
                "Quantification: Include specific numbers and metrics in achievements",
                "Action Verbs: Replace weak verbs with strong action words",
                "Structure: Improve formatting for better ATS readability"
            ]
        elif score < 80:
            return [
                "Keywords: Emphasize your most relevant experiences",
                "Quantification: Add specific examples demonstrating required skills",
                "Structure: Tailor your summary to better match the role"
            ]
        else:
            return [
                "Keywords: Add any remaining missing keywords for ATS optimization",
                "Structure: Ensure contact information is current and complete",
                "Quantification: Highlight your strongest achievements prominently"
            ]

def generate_skill_gap_analysis(resume_text, job_requirement):
    """Generate skill gap analysis with learning resources"""
    try:
        prompt = f"""
        As a career development expert, analyze the skill gaps between this resume and job requirements.
        
        Resume: {resume_text[:1000]}...
        Job Requirements: {job_requirement[:800]}...
        
        Identify:
        1. Current Skills: List 5-8 key skills found in the resume
        2. Skill Gaps: List 3-5 important skills missing from resume but required for the job
        3. Learning Resources: For each skill gap, suggest 1-2 specific online courses, certifications, or resources
        
        Format as JSON:
        {
            "current_skills": ["skill1", "skill2", ...],
            "skill_gaps": [
                {
                    "skill": "skill_name",
                    "importance": "high/medium/low",
                    "resources": ["resource1", "resource2"]
                }
            ]
        }
        """
        
        response = gemini_model.generate_content(prompt)
        
        # Try to extract JSON from response
        import json
        try:
            # Look for JSON in the response
            response_text = response.text.strip()
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback structure
        return {
            "current_skills": ["Communication", "Problem Solving", "Team Collaboration", "Project Management"],
            "skill_gaps": [
                {
                    "skill": "Advanced Analytics",
                    "importance": "high",
                    "resources": ["Coursera Data Analytics Certificate", "Google Analytics Academy"]
                },
                {
                    "skill": "Cloud Computing",
                    "importance": "medium", 
                    "resources": ["AWS Cloud Practitioner", "Azure Fundamentals"]
                }
            ]
        }
        
    except Exception as e:
        return {
            "current_skills": ["Communication", "Problem Solving", "Team Collaboration"],
            "skill_gaps": [
                {
                    "skill": "Technical Skills",
                    "importance": "high",
                    "resources": ["Online courses", "Professional certifications"]
                }
            ]
        }

def summarize_text(text, num_sentences=3):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return " ".join(sentences[:num_sentences])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
