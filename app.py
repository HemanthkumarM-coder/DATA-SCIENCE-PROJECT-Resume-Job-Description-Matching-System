from flask import Flask, render_template, request, redirect, flash, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from utils import calculate_similarity_batch, analyze_skills

app = Flask(__name__)
app.secret_key = "super_secret_key"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resumes' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    files = request.files.getlist('resumes')
    if not files or all(file.filename == '' for file in files):
        flash('No selected file')
        return redirect(url_for('index'))
    
    job_description = request.form.get('job_description')
    
    if not job_description:
        flash('Please provide a job description')
        return redirect(url_for('index'))
        
    try:
        from utils import extract_text_from_pdf
        
        parsed_data = []
        for file in files:
            if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                text = ""
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                else:
                    # It's a txt file
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        
                parsed_data.append({
                    'Filename': filename,
                    'Resume_Text': text
                })
        
        if not parsed_data:
            flash('No valid PDF or TXT files were processed.')
            return redirect(url_for('index'))
            
        df = pd.DataFrame(parsed_data)
        
        # Process Data
        scores = calculate_similarity_batch(df['Resume_Text'], job_description)
        df['Similarity_Score'] = scores
        
        # Sort by similarity to get the best match
        df_sorted = df.sort_values(by='Similarity_Score', ascending=False)
        
        # Focus on the top candidate
        top_row = df_sorted.iloc[0]
        top_resume_text = str(top_row['Resume_Text'])
        top_candidate_name = str(top_row['Filename'])
        similarity_score = top_row['Similarity_Score']
                
        # Use existing advanced analysis logic
        _, strong_matches, missing_skills, weak_skills = analyze_skills(top_resume_text, job_description)
        
        return render_template(
            'results.html', 
            top_candidate_name=top_candidate_name,
            similarity_score=similarity_score,
            strong_matches=strong_matches,
            weak_skills=weak_skills,
            missing_skills=missing_skills
        )
        
    except Exception as e:
        flash(f"Error processing files: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
