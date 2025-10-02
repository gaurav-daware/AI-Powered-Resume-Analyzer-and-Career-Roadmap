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
detected_domain = None

# --- Domain-Specific Skills Dictionary ---
DOMAIN_SKILLS = {
    "software_engineering": [
        "Python", "Java", "C++", "JavaScript", "PHP", "SQL", "MySQL", "MongoDB", "SQLite", "BigQuery",
        "PostgreSQL", "HTML", "CSS", "Dart", "R", "Ruby", "Swift", "Kotlin", "TypeScript",
        "NumPy", "Pandas", "SciPy", "Dask", "GeoPandas", "Sklearn", "NLTK", "OpenCV", "Keras",
        "TensorFlow", "Pytorch", "AzureML", "Matplotlib", "Seaborn", "Plotly", "Flask", "Django",
        "Node.js", "React", "Flutter", "Bootstrap", "CodeIgniter", "REST API", "WebRTC", "MLOps",
        "Docker", "Kubernetes", "Kubeflow", "AWS", "GCP", "Azure", "Google Colab", "Heroku",
        "Jupyter", "Git", "Github", "VSCode", "Machine Learning", "Deep Learning", "NLP",
        "Computer Vision", "Data Engineering", "IoT", "Cloud Computing", "Scrum",
        "Agile", "Spark", "Hadoop", "Kafka", "Data Science", "Data Analytics", "CI/CD",
        "Microservices", "GraphQL", "Redis", "Elasticsearch", "Jenkins", "Terraform"
    ],
    "mba_business": [
        "Leadership", "Strategy", "Operations", "Marketing", "Finance", "Project Management",
        "Excel", "SAP", "Six Sigma", "Lean", "Business Analysis", "Strategic Planning",
        "Financial Analysis", "Budget Management", "P&L", "ROI", "KPI", "Stakeholder Management",
        "Change Management", "Risk Management", "Business Development", "Sales", "CRM",
        "Market Research", "Product Management", "Supply Chain", "Procurement", "Negotiation",
        "Corporate Strategy", "M&A", "Consulting", "PowerPoint", "Tableau", "Power BI",
        "ERP", "Salesforce", "Analytics", "Digital Marketing", "Brand Management"
    ],
    "pharmacy_healthcare": [
        "Pharmacology", "Clinical Trials", "Regulatory Compliance", "FDA", "GMP", "GLP",
        "Research Methods", "Lab Techniques", "Drug Development", "Patient Care",
        "Pharmaceutical Sciences", "Clinical Research", "CAPA", "SOP", "Quality Assurance",
        "Quality Control", "ICH Guidelines", "Pharmacy Practice", "Medication Management",
        "Clinical Pharmacology", "Pharmacokinetics", "Pharmacodynamics", "Drug Information",
        "Compounding", "Dispensing", "Patient Counseling", "Healthcare Systems", "EHR",
        "Medical Terminology", "Anatomy", "Physiology", "Pathology", "Therapeutics",
        "Biostatistics", "Epidemiology", "Protocol Development"
    ],
    "law_legal": [
        "Legal Research", "Litigation", "Contract Drafting", "Arbitration", "Compliance",
        "Legal Writing", "Case Management", "Due Diligence", "Regulatory Law", "Corporate Law",
        "Intellectual Property", "Patent Law", "Tax Law", "Employment Law", "Real Estate Law",
        "Mergers and Acquisitions", "Legal Analysis", "Negotiation", "Mediation", "Discovery",
        "Westlaw", "LexisNexis", "Legal Brief", "Memorandum", "Deposition", "Trial Preparation",
        "Contract Negotiation", "Risk Assessment", "Privacy Law", "Data Protection", "GDPR",
        "Securities Law", "International Law", "Criminal Law", "Civil Litigation"
    ],
    "marketing_creative": [
        "Digital Marketing", "Content Marketing", "SEO", "SEM", "Social Media Marketing",
        "Google Analytics", "Google Ads", "Facebook Ads", "Email Marketing", "Marketing Strategy",
        "Brand Management", "Campaign Management", "Copywriting", "Creative Strategy",
        "Market Research", "Consumer Insights", "Adobe Creative Suite", "Photoshop", "Illustrator",
        "InDesign", "Video Production", "Content Creation", "Influencer Marketing",
        "Marketing Automation", "HubSpot", "Mailchimp", "A/B Testing", "Conversion Optimization",
        "Marketing Analytics", "CRM", "Public Relations", "Event Marketing", "Product Marketing"
    ],
    "finance_accounting": [
        "Financial Analysis", "Accounting", "Bookkeeping", "Financial Reporting", "Auditing",
        "Tax Preparation", "GAAP", "IFRS", "CPA", "Financial Modeling", "Budgeting", "Forecasting",
        "QuickBooks", "SAP", "Oracle", "Excel", "VBA", "SQL", "Bloomberg", "Reuters",
        "Investment Analysis", "Portfolio Management", "Risk Management", "Corporate Finance",
        "Valuation", "M&A", "Treasury", "Cash Flow Management", "Cost Accounting",
        "Management Accounting", "Financial Planning", "Compliance", "Internal Controls"
    ],
    "generic": [
        "Communication", "Teamwork", "Problem Solving", "Adaptability", "Critical Thinking",
        "Time Management", "Organization", "Attention to Detail", "Customer Service",
        "Interpersonal Skills", "Collaboration", "Flexibility", "Initiative", "Work Ethic",
        "Analytical Skills", "Decision Making", "Conflict Resolution", "Presentation Skills",
        "Research", "Writing", "Public Speaking", "Multitasking", "Creativity"
    ]
}

def get_domain_skills(domain):
    """Get skills for a specific domain including generic skills."""
    domain_key = domain.lower().replace(" ", "_").replace("/", "_")
    skills = DOMAIN_SKILLS.get(domain_key, [])
    generic = DOMAIN_SKILLS.get("generic", [])
    return skills + generic

def create_matcher_for_domain(domain):
    """Create a spaCy matcher with domain-specific skills."""
    matcher = Matcher(nlp.vocab)
    skills = get_domain_skills(domain)
    
    for skill in skills:
        # Handle multi-word skills
        words = skill.split()
        if len(words) == 1:
            pattern = [{"LOWER": skill.lower()}]
        else:
            pattern = [{"LOWER": word.lower()} for word in words]
        matcher.add(skill, [pattern])
    
    return matcher

def detect_resume_domain(text):
    """Automatically detect the career domain of the resume."""
    try:
        prompt = f"""
        Analyze this resume and classify it into ONE of the following domains:
        - Software Engineering
        - MBA Business
        - Pharmacy Healthcare
        - Law Legal
        - Marketing Creative
        - Finance Accounting
        - Other
        
        Resume Text: {text[:1500]}...
        
        Respond with ONLY the domain name exactly as shown above (e.g., "Software Engineering").
        """
        
        response = gemini_model.generate_content(prompt)
        domain = response.text.strip()
        
        # Validate and normalize domain
        valid_domains = [
            "Software Engineering", "MBA Business", "Pharmacy Healthcare",
            "Law Legal", "Marketing Creative", "Finance Accounting", "Other"
        ]
        
        for valid_domain in valid_domains:
            if valid_domain.lower() in domain.lower():
                return valid_domain
        
        return "Other"
        
    except Exception as e:
        print(f"Error detecting domain: {e}")
        return "Other"

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
    global current_resume, cleaned_resume_text, detected_domain
    
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
            
            # Detect domain automatically
            detected_domain = detect_resume_domain(cleaned_resume_text)

            response = {
                "message": "Resume uploaded and processed successfully",
                "filename": filename,
                "parsed_text_length": len(cleaned_resume_text),
                "detected_domain": detected_domain
            }
            return jsonify(response), 201
            
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

@app.route('/delete', methods=['POST'])
def delete_resume():
    global current_resume, cleaned_resume_text, detected_domain
    
    if current_resume is None:
        return jsonify({"message": "No resume to delete"}), 404

    filename = current_resume[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    
    current_resume = None
    cleaned_resume_text = None
    detected_domain = None
    
    return jsonify({"message": f"File '{filename}' deleted successfully"}), 200

def delete_all_files():
    """Deletes all files in the upload and highlighted folders."""
    for folder in [UPLOAD_FOLDER, app.config['HIGHLIGHT_FOLDER']]:
        if os.path.exists(folder):
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
            f"You are an expert career advisor specializing in {detected_domain}. "
            f"Based on the following resume text, provide a career roadmap and suggestions. "
            f"The user is asking: '{user_query}'\n\n"
            f"Resume Text:\n{cleaned_resume_text}\n\n"
            f"Suggestions should include skill gaps, recommended courses/certifications specific to {detected_domain}, "
            f"and potential next career steps in this field. Format your response clearly in markdown."
        )
        
        response = gemini_model.generate_content(prompt)
        
        return jsonify({
            "response": response.text,
            "domain": detected_domain
        }), 200

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

    analysis_result = perform_detailed_analysis(cleaned_resume_text, job_requirement, detected_domain)
    
    response = {
        "job_requirement": job_requirement,
        "resume_count": 1,
        "detected_domain": detected_domain,
        "results": [{
            "filename": current_resume[0],
            **analysis_result
        }]
    }
    return jsonify(response), 200

def perform_detailed_analysis(resume_text, job_requirement, domain):
    """Perform comprehensive resume analysis with domain-specific scoring."""
    
    # Create domain-specific matcher
    matcher = create_matcher_for_domain(domain)
    
    job_doc = nlp(job_requirement.lower())
    resume_doc = nlp(resume_text.lower())
    
    job_keywords = set([token.text for token in job_doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2])
    resume_keywords = set([token.text for token in resume_doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2])

    job_skills = set([nlp.vocab.strings[match_id] for match_id, start, end in matcher(job_doc)])
    resume_skills = set([nlp.vocab.strings[match_id] for match_id, start, end in matcher(resume_doc)])

    # Combined keywords for TF-IDF
    job_combined = " ".join(job_keywords.union(job_skills))
    resume_combined = " ".join(resume_keywords.union(resume_skills))

    documents = [resume_combined, job_combined]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    base_score = cosine_sim[0][0] * 100
    
    # Calculate keyword matches
    matching_keywords = list(job_keywords.intersection(resume_keywords))
    matching_skills = list(job_skills.intersection(resume_skills))
    
    all_matches = list(set(matching_keywords + matching_skills))
    
    # Calculate a weighted score
    keyword_weight = 0.6
    skill_weight = 0.4
    
    weighted_score = (base_score * keyword_weight * 10)
    if job_skills:
        skill_match_percentage = len(matching_skills) / len(job_skills)
        weighted_score += (skill_match_percentage * 100 * skill_weight)
    
    final_score = min(100, weighted_score)
    
    # Missing keywords
    missing_keywords = list(job_keywords - resume_keywords)
    missing_skills = list(job_skills - resume_skills)
    all_missing = list(set(missing_keywords + missing_skills))

    # Calculate keyword density for top skills
    keyword_density = {}
    for keyword in all_matches:
        count = resume_text.lower().count(keyword.lower())
        keyword_density[keyword] = count
    
    recommendations = generate_recommendations(resume_text, job_requirement, final_score, all_missing, domain)
    
    skill_gaps = generate_skill_gap_analysis(resume_text, job_requirement, domain)
    
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
    """Get ATS compatibility status based on score."""
    if score >= 85:
        return {"level": "high", "label": "Excellent Match", "color": "green"}
    elif score >= 70:
        return {"level": "medium", "label": "Strong Match", "color": "yellow"}
    elif score >= 50:
        return {"level": "medium", "label": "Good Match", "color": "orange"}
    else:
        return {"level": "low", "label": "Needs Significant Improvement", "color": "red"}

def generate_recommendations(resume_text, job_requirement, score, missing_keywords, domain):
    """Generate AI-powered domain-specific recommendations."""
    try:
        domain_guidance = {
            "Software Engineering": "Focus on technical skills, projects, GitHub contributions, and quantifiable achievements in software development.",
            "MBA Business": "Emphasize leadership, strategic initiatives, financial impact (ROI, cost savings), and team management.",
            "Pharmacy Healthcare": "Highlight clinical experience, regulatory compliance (FDA, GMP), research contributions, and patient care outcomes.",
            "Law Legal": "Showcase case experience, legal research skills, publications, notable cases, and areas of specialization.",
            "Marketing Creative": "Demonstrate campaign results, creative projects, metrics (engagement, conversions), and portfolio work.",
            "Finance Accounting": "Include financial modeling, audit experience, certifications (CPA, CFA), and quantifiable financial achievements."
        }
        
        guidance = domain_guidance.get(domain, "Focus on relevant skills and quantifiable achievements in your field.")
        
        prompt = f"""
        As an expert resume consultant specializing in {domain}, analyze this resume against the job requirements and provide specific, actionable recommendations to improve its ATS score.

        Domain-Specific Guidance: {guidance}
        Resume Score: {score:.1f}%
        Missing Keywords: {', '.join(missing_keywords[:5])}

        Resume Text: {resume_text[:1500]}...
        Job Requirements: {job_requirement[:1000]}...

        Provide exactly 4-6 specific recommendations in this format:
        1. [Category]: [Specific actionable advice]

        Categories should be relevant to {domain}, like: Keywords, Quantification, Action Verbs, Formatting, Structure, Skills, Experience, Certifications, Projects.
        Keep each recommendation under 120 characters and prioritize the most impactful changes for this domain.
        """
        
        response = gemini_model.generate_content(prompt)
        recommendations_text = response.text.strip()
        
        recommendations = [line.strip() for line in recommendations_text.split('\n') if line.strip() and line.strip()[0].isdigit()]
        
        return recommendations[:6]
        
    except Exception as e:
        print(f"Error generating AI recommendations: {e}")
        return [
            "Keywords: Integrate missing keywords from the job description.",
            "Quantification: Add numbers and metrics to your accomplishments.",
            "Action Verbs: Start bullet points with strong action verbs.",
            "Structure: Ensure your resume sections are clearly labeled."
        ]

def generate_skill_gap_analysis(resume_text, job_requirement, domain):
    """Generate domain-specific skill gap analysis with learning resources."""
    try:
        domain_resources = {
            "Software Engineering": "Focus on technical courses (Udemy, Coursera), certifications (AWS, Google Cloud), and coding platforms (LeetCode, HackerRank).",
            "MBA Business": "Suggest business courses (Coursera MBA, edX), certifications (PMP, Six Sigma), and leadership programs.",
            "Pharmacy Healthcare": "Include pharmacy certifications, FDA training, clinical research courses, and healthcare compliance programs.",
            "Law Legal": "Recommend legal research tools training (Westlaw, LexisNexis), bar review courses, and specialized legal certifications.",
            "Marketing Creative": "Suggest digital marketing certifications (Google Analytics, HubSpot), creative tools training (Adobe), and marketing courses.",
            "Finance Accounting": "Include CPA/CFA prep, financial modeling courses, and accounting software certifications (QuickBooks, SAP)."
        }
        
        resource_guidance = domain_resources.get(domain, "Suggest relevant courses and certifications for this field.")
        
        prompt = f"""
        As a career development expert specializing in {domain}, analyze the skill gaps between this resume and the job requirements.
        
        {resource_guidance}
        
        Resume: {resume_text[:1000]}...
        Job Requirements: {job_requirement[:800]}...
        
        Identify:
        1. Current Skills: List 5-8 key skills from the resume relevant to {domain}.
        2. Skill Gaps: List 3-5 important skills missing but required for the job in {domain}.
        3. Learning Resources: For each skill gap, suggest 1-2 specific resources appropriate for {domain} (courses, certifications, platforms).
        
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
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error generating skill gap analysis: {e}")
        return {
            "current_skills": ["Communication", "Teamwork"],
            "skill_gaps": [
                {
                    "skill": "Domain-Specific Skills",
                    "importance": "high",
                    "resources": ["Relevant Online Course", "Professional Certification"]
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