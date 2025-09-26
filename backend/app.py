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
import json

# --- Load environment variables for API key ---
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Initialize with Google's Gemini API ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

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

# --- Improved Tech Entities List ---
TECH_ENTITIES = [
    "Python", "Java", "C++", "JavaScript", "PHP", "SQL", "MySQL", "MongoDB", "SQLite", "BigQuery",
    "PostgreSQL", "HTML", "CSS", "Dart", "R", "Ruby", "Swift", "Kotlin", "TypeScript",
    "NumPy", "Pandas", "SciPy", "Dask", "GeoPandas", "Sklearn", "NLTK", "OpenCV", "Keras",
    "TensorFlow", "Pytorch", "AzureML", "Matplotlib", "Seaborn", "Plotly", "Flask", "Django",
    "Node.js", "React", "Flutter", "Bootstrap", "CodeIgniter", "REST API", "WebRTC", "MLOps",
    "Docker", "Kubernetes", "Kubeflow", "AWS", "GCP", "Azure", "Google Colab", "Heroku",
    "Jupyter", "Git", "Github", "VSCode", "Machine Learning", "Deep Learning", "NLP",
    "Computer Vision", "Data Engineering", "IoT", "Cloud Computing", "AutoCAD", "Scrum",
    "Agile", "Tableau", "Power BI", "Spark", "Hadoop", "Kafka", "Data Science", "Data Analytics"
]

matcher = Matcher(nlp.vocab)
for tech in TECH_ENTITIES:
    pattern = [{"LOWER": tech.lower()}]
    matcher.add(tech, [pattern])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_single_resume(text):
    if not text.strip():
        return None
    cleaned_text = re.sub(r'[\uf000-\uf8ff]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

@app.route('/upload', methods=['POST'])
def upload_resume():
    global current_resume, cleaned_resume_text
    
    delete_all_files()

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            if not text.strip():
                return jsonify({"error": "Could not extract text from PDF. Please ensure it's a valid PDF with readable text."}), 400
            
            cleaned_resume_text = process_single_resume(text)
            current_resume = (filename, text.strip())

            response = {
                "message": "Resume uploaded and processed successfully",
                "filename": filename,
                "parsed_text_length": len(cleaned_resume_text)
            }
            return jsonify(response), 201
            
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

@app.route('/delete', methods=['POST'])
def delete_resume():
    global current_resume, cleaned_resume_text
    
    if current_resume is None:
        return jsonify({"message": "No resume to delete"}), 404

    filename = current_resume[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    
    current_resume = None
    cleaned_resume_text = None
    
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
            f"and potential next career steps. Format your response clearly in markdown."
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
    """Perform comprehensive resume analysis with a more refined scoring model."""
    
    job_doc = nlp(job_requirement.lower())
    resume_doc = nlp(resume_text.lower())
    
    job_keywords = set([token.text for token in job_doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2])
    resume_keywords = set([token.text for token in resume_doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2])

    job_techs = set([nlp.vocab.strings[match_id] for match_id, start, end in matcher(job_doc)])
    resume_techs = set([nlp.vocab.strings[match_id] for match_id, start, end in matcher(resume_doc)])

    # Combined keywords for TF-IDF
    job_combined = " ".join(job_keywords.union(job_techs))
    resume_combined = " ".join(resume_keywords.union(resume_techs))

    documents = [resume_combined, job_combined]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    base_score = cosine_sim[0][0] * 100
    
    # Calculate keyword matches
    matching_keywords = list(job_keywords.intersection(resume_keywords))
    matching_techs = list(job_techs.intersection(resume_techs))
    
    all_matches = list(set(matching_keywords + matching_techs))
    
    # Calculate a weighted score
    keyword_weight = 0.6
    tech_weight = 0.4
    
    weighted_score = (base_score * keyword_weight *10)
    if job_techs:
        tech_match_percentage = len(matching_techs) / len(job_techs)
        weighted_score += (tech_match_percentage * 100 * tech_weight)
    
    final_score = min(100, weighted_score)
    
    # Missing keywords
    missing_keywords = list(job_keywords - resume_keywords)
    missing_techs = list(job_techs - resume_techs)
    all_missing = list(set(missing_keywords + missing_techs))

    # Calculate keyword density for top skills
    keyword_density = {}
    for keyword in all_matches:
        count = resume_text.lower().count(keyword.lower())
        keyword_density[keyword] = count
    
    recommendations = generate_recommendations(resume_text, job_requirement, final_score, all_missing)
    
    skill_gaps = generate_skill_gap_analysis(resume_text, job_requirement)
    
    return {
        "score": round(final_score, 2),
        "summary": summarize_text(resume_text),
        "ats_status": get_ats_status(final_score),
        "keyword_analysis": {
            "matching_keywords": all_matches[:15],
            "missing_keywords": all_missing[:10],
            "keyword_density": dict(sorted(keyword_density.items(), key=lambda x: x[1], reverse=True)[:8])
        },
        "recommendations": recommendations,
        "skill_gaps": skill_gaps
    }

def get_ats_status(score):
    """Get ATS compatibility status based on the new, improved score."""
    if score >= 85:
        return {"level": "high", "label": "Excellent Match", "color": "green"}
    elif score >= 70:
        return {"level": "medium", "label": "Strong Match", "color": "yellow"}
    elif score >= 50:
        return {"level": "medium", "label": "Good Match", "color": "orange"}
    else:
        return {"level": "low", "label": "Needs Significant Improvement", "color": "red"}

def generate_recommendations(resume_text, job_requirement, score, missing_keywords):
    """Generate AI-powered recommendations with a more specific prompt."""
    try:
        prompt = f"""
        As an expert resume consultant, analyze this resume against the job requirements and provide specific, actionable recommendations to improve its ATS score.

        Resume Score: {score:.1f}%
        Missing Keywords: {', '.join(missing_keywords[:5])}

        Resume Text: {resume_text[:1500]}...
        Job Requirements: {job_requirement[:1000]}...

        Provide exactly 4-6 specific recommendations in this format:
        1. [Category]: [Specific actionable advice]

        Categories should be concise, like: Keywords, Quantification, Action Verbs, Formatting, Structure, Skills, Experience.
        Keep each recommendation under 120 characters and prioritize the most impactful changes.
        """
        
        response = gemini_model.generate_content(prompt)
        recommendations_text = response.text.strip()
        
        recommendations = [line.strip() for line in recommendations_text.split('\n') if line.strip().startswith(tuple('123456'))]
        
        return recommendations[:6]
        
    except Exception as e:
        print(f"Error generating AI recommendations: {e}")
        if score < 60:
            return [
                "Keywords: Integrate missing keywords from the job description.",
                "Quantification: Add numbers and metrics to your accomplishments.",
                "Action Verbs: Start bullet points with strong action verbs.",
                "Structure: Ensure your resume sections are clearly labeled."
            ]
        elif score < 80:
            return [
                "Keywords: Tailor your summary and skills section to the role.",
                "Quantification: Provide more specific data points for key projects.",
                "Experience: Connect your experience more explicitly to job duties.",
                "Formatting: Check for clean, readable formatting for ATS."
            ]
        else:
            return [
                "Keywords: Review your resume for any minor keyword gaps.",
                "Structure: Fine-tune your professional summary for a perfect fit.",
                "Quantification: Highlight the most impressive metrics in your top achievements."
            ]

def generate_skill_gap_analysis(resume_text, job_requirement):
    """Generate skill gap analysis with learning resources, formatted as JSON."""
    try:
        prompt = f"""
        As a career development expert, analyze the skill gaps between this resume and the job requirements.
        
        Resume: {resume_text[:1000]}...
        Job Requirements: {job_requirement[:800]}...
        
        Identify:
        1. Current Skills: List 5-8 key skills from the resume.
        2. Skill Gaps: List 3-5 important skills missing but required for the job.
        3. Learning Resources: For each skill gap, suggest 1-2 specific online courses, certifications, or resources (e.g., Coursera, Udacity, AWS certification).
        
        Format your response as a single, valid JSON object with the keys "current_skills" and "skill_gaps".
        Example:
        {{
            "current_skills": ["skill1", "skill2"],
            "skill_gaps": [
                {{
                    "skill": "skill_name",
                    "importance": "high/medium/low",
                    "resources": ["resource1", "resource2"]
                }}
            ]
        }}
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '')
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error generating skill gap analysis: {e}")
        return {
            "current_skills": ["Communication", "Teamwork"],
            "skill_gaps": [
                {
                    "skill": "Advanced Technical Skills",
                    "importance": "high",
                    "resources": ["Udemy Course", "Coursera Specialization"]
                }
            ]
        }

def summarize_text(text, num_sentences=3):
    """Creates a brief summary of the resume text."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return " ".join(sentences[:num_sentences])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)