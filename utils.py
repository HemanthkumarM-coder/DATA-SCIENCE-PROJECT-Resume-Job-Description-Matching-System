import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# A massive list of common IT / Data Science / Dev skills for extraction, unigrams and bigrams
COMMON_SKILLS = [
    'python', 'java', 'c++', 'sql', 'nosql', 'machine learning', 'deep learning',
    'data analysis', 'data science', 'nlp', 'natural language processing', 'flask',
    'django', 'streamlit', 'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'html', 
    'css', 'javascript', 'react', 'vue', 'angular', 'git', 'linux', 'statistics',
    'mathematics', 'communication', 'problem solving', 'tensorflow', 'pytorch',
    'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 
    'power bi', 'excel', 'agile', 'scrum', 'backend', 'frontend', 'full stack',
    'rest api', 'graphql', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust',
    'apache spark', 'hadoop', 'kafka', 'ci/cd', 'jenkins', 'github actions',
    'data visualization', 'data mining', 'predictive modeling', 'artificial intelligence',
    'computer vision', 'opencv', 'generative ai', 'llm', 'chatgpt', 'prompt engineering',
    'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'cassandra',
    'snowflake', 'redshift', 'bigquery', 'dbt', 'airflow', 'luigi', 'fastapi',
    'spring boot', 'express.js', 'node.js', 'next.js', 'tailwind css', 'bootstrap',
    'figma', 'ui/ux', 'product management', 'project management', 'jira', 'confluence',
    'ansible', 'terraform', 'cloudformation', 'linux administration', 'bash scripting',
    'powershell', 'cybersecurity', 'penetration testing', 'cryptography',
    'time series analysis', 'ab testing', 'hypothesis testing', 'regression analysis',
    'classification', 'clustering', 'decision trees', 'random forest', 'xgboost',
    'lightgbm', 'catboost', 'svm', 'neural networks', 'cnn', 'rnn', 'lstm', 'transformer',
    'bert', 'gpt', 'hugging face', 'mlops', 'model deployment', 'sagemaker',
    'vertex ai', 'data engineering', 'etl', 'elt', 'data warehousing', 'data lakes',
    'data modeling', 'scala', 'r', 'julia', 'matlab', 'perl', 'shell', 'typescript',
    'dart', 'flutter', 'react native', 'ionic', 'xamarin', 'android', 'ios'
]

def extract_text_from_pdf(filepath):
    """Extracts text content from a PDF file."""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return text

def clean_text(text):
    """Preprocess text by lowercasing, removing punctuation, and dropping basic stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def calculate_similarity_batch(resumes_series, job_desc_text):
    """
    Calculates a hybrid match score between the Job Description and a batch of resumes.
    Combines:
    1. TF-IDF Cosine Similarity (40% weight) - Captures general context and phrasing.
    2. Skill Match Ratio (60% weight) - Captures explicit keyword/competency alignment.
    """
    # 1. General TF-IDF Cosine Similarity
    cleaned_jd = clean_text(job_desc_text)
    cleaned_resumes = resumes_series.apply(clean_text)
    
    all_texts = [cleaned_jd] + cleaned_resumes.tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # 2. Skill-Specific Match Ratio
    # Extract unique skill sets from JD
    jd_skills = set(extract_skills_with_frequency(job_desc_text).keys())
    
    hybrid_scores = []
    for i, resume_text in enumerate(resumes_series):
        res_skills = set(extract_skills_with_frequency(resume_text).keys())
        
        # Calculate ratio of JD skills found in resume
        if len(jd_skills) > 0:
            skill_match_ratio = len(jd_skills.intersection(res_skills)) / len(jd_skills)
        else:
            skill_match_ratio = 0
            
        # Weighted Final Score: 40% Context, 60% Specific Skills
        text_alignment = cosine_similarities[i]
        final_score = (0.4 * text_alignment) + (0.6 * skill_match_ratio)
        
        # Cap at 100% and round
        hybrid_scores.append(round(min(final_score * 100, 100), 2))
        
    return hybrid_scores

def extract_skills_with_frequency(text, skills_list=COMMON_SKILLS):
    """Extract skills from text and map their frequency count."""
    text_cleaned = clean_text(text)
    skill_counts = Counter()
    
    for skill in skills_list:
        # Check if the skill word/phrase exists in the cleaned text 
        # using a case-insensitive boundary match.
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text_cleaned))
        if count > 0:
            skill_counts[skill] = count
            
    return dict(skill_counts)

def analyze_skills(resume_text, job_desc_text):
    """
    Returns dictionaries of: matches, missing_skills, and weak_skills.
    A skill is weak if it is required by the JD, found in the resume, 
    but its frequency in the resume is distinctly much lower.
    """
    jd_counts = extract_skills_with_frequency(job_desc_text)
    resume_counts = extract_skills_with_frequency(resume_text)
    
    jd_skills = set(jd_counts.keys())
    resume_skills = set(resume_counts.keys())
    
    missing_raw = jd_skills - resume_skills
    matched_skills = jd_skills.intersection(resume_skills)
    
    missing_skills = []
    for skill in missing_raw:
        jd_freq = jd_counts[skill]
        priority = "High" if jd_freq > 2 else "Standard"
        
        # Professional importance description based on JD frequency
        if priority == "High":
            importance = "Critical requirement for this role. Absence significantly impacts candidate eligibility."
        else:
            importance = "Standard requirement. Including this would strengthen professional alignment with the role."

        # Improvement Path description
        improvement_path = f"Integrating {skill} into your profile will demonstrate competency in a core requirement, allowing you to execute higher-level tasks and align with the technical expectations for this specific position."

        missing_skills.append({
            'name': skill,
            'reason': importance,
            'improvement': improvement_path,
            'priority': priority
        })
    
    weak_skills = []
    strong_matches = []
    
    # Analyze the matched skills for frequency strength
    for skill in matched_skills:
        jd_freq = jd_counts[skill]
        res_freq = resume_counts[skill]
        
        # If the skill is heavily emphasized in JD, but barely mentioned in Resume
        if jd_freq > 1 and res_freq == 1:
            weak_skills.append({
                'name': skill,
                'reason': f"Highly requested (appears {jd_freq} times in JD), but only mentioned once in your resume.",
                'details': f"Consider adding more specific projects or experience related to {skill}."
            })
        else:
            strong_matches.append(skill)
            
    # Also add extraneous resume skills that aren't in the JD to candidate skills list
    all_candidate_skills = list(resume_skills)
            
    return all_candidate_skills, strong_matches, missing_skills, weak_skills
