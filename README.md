# Resume Matcher Professional

An enterprise-grade candidate analysis platform that uses a hybrid similarity model to match resumes with job descriptions.

## Features

- **Hybrid Similarity Scoring**: Combines TF-IDF text alignment (40%) with explicit skill coverage (60%) for high-accuracy matching.
- **Single-Page Analysis**: Comprehensive view of Core Matches, Optimization Areas, and Critical Gaps on a single dashboard.
- **Deep Gap Analysis**: Detailed "Impact Analysis" for each missing skill, explaining how acquiring the skill improves professional candidacy.
- **Premium UI/UX**: Professional Slate-Indigo design system with high-quality typography (Inter) and dashboard visualization.
- **Multi-Format Support**: Supports both PDF and TXT file uploads.

## Project Structure

- `app.py`: Main Flask application and routing.
- `utils.py`: Core logic for text processing, hybrid scoring, and skill analysis.
- `static/style.css`: Premium CSS styling and design system tokens.
- `templates/`: Jinja2 HTML templates for the landing and results pages.
- `requirements.txt`: Python package dependencies.
- `test_script.py`: CLI-based verification script for the matching logic.

## Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Interface**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

4. **Analyze**:
   Upload candidate resumes (PDF/TXT) and paste your targeting job requirements to see a detailed analysis.

## Technology Stack

- **Backend**: Python, Flask, Pandas
- **AI/ML**: Scikit-learn (TF-IDF, Cosine Similarity)
- **Parsing**: PyPDF2
- **Frontend**: HTML5, Vanilla CSS, Font Awesome
- **Design**: Slate & Indigo Theme, Inter Typography
