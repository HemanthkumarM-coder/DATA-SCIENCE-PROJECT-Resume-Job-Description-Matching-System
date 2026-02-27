import pandas as pd
from utils import calculate_similarity_batch, extract_skills, get_missing_skills

# Load mock data
df = pd.read_csv('mock_resumes.csv')
print("Successfully loaded CSV.")

job_desc = "We are looking for a Data Scientist or Machine Learning Engineer with strong Python and SQL skills. Experience with TensorFlow and PyTorch is a huge plus."

# Test similarity
scores = calculate_similarity_batch(df['Resume_Text'], job_desc)
df['Score'] = scores
print("\nSimilarity Scores:")
print(df[['Candidate_Name', 'Score']].sort_values(by='Score', ascending=False))

# Test skills extraction
top_resume = df.sort_values(by='Score', ascending=False).iloc[0]['Resume_Text']

# Use the advanced analyze_skills 
all_skills, strong, missing, weak = analyze_skills(top_resume, job_desc)

print("\n--- SKILL ANALYSIS For Top Candidate ---")
print(f"Strong Matches: {strong}")
print(f"Weak Skills: {[s['name'] for s in weak]}")
print(f"Missing Skills: {[s['name'] for s in missing]}")
