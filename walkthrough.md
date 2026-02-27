# 📄 Resume Matcher: Semantic AI Analysis

A sophisticated recruitment tool that analyzes the semantic similarity between Resumes and Job Descriptions using NLP.

## 📋 Features
- **Cosine Similarity Scoring**: Accurate matching using advanced text vectorization.
- **Skill Gap Analysis**: Separates skills into "Critical" (Matched), "Weak" (Frequent in JD, sparse in Resume), and "Missing".
- **Quantitative Insights**: Displays exact mention counts for every skill across both documents.
- **Actionable Feedback**: Explains why certain skills are flagged as critical versus weak.
- **Responsive Web UI**: Premium "Indigo-Slate" dashboard with glassmorphism alerts.

## ⚙️ Technical Workflow
1. **Extraction**: PDF/Docx text parsing.
2. **Preprocessing**: Tokenization and removal of stop words.
3. **Semantic Scoring**: Implementation of Cosine Similarity via Scikit-Learn.
4. **Logic Engine**: Priority-based skill categorization.
5. **Presentation**: Dynamic HTML serving via Flask.

## 🚀 How to Run
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000` to start matching.
