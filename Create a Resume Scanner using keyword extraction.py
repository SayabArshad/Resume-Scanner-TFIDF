#import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resumes and job description
resumes = {
    'resume_id': [1, 2, 3, 4],
    'resume_text': [
        "Experienced software developer with expertise in Python, Java, and machine learning.",
        "Project manager with a strong background in Agile methodologies and team leadership.",
        "Data scientist skilled in data analysis, statistical modeling, and Python programming.",
        "Marketing specialist with experience in digital marketing, SEO, and content creation."
    ]
}

job_description = "Looking for a software developer proficient in Python and machine learning to join our dynamic team."

# Create a DataFrame for resumes
df_resumes = pd.DataFrame(resumes)
print("Resumes DataSet:\n", df_resumes)

# Combine resumes and job description for vectorization
documents = df_resumes['resume_text'].tolist() + [job_description]  # REMOVED THE DUPLICATE APPEND

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity between resumes and job description
# tfidf_matrix[:-1] = all resumes (first 4 documents)
# tfidf_matrix[-1:] = job description (last document)
cosine_sim = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])

# Add similarity scores to dataframe
df_resumes['similarity_score'] = cosine_sim.flatten()
print("\nResumes with Similarity Scores:\n", df_resumes)

# Identify top matching resumes
threshold = 0.2
top_resumes = df_resumes[df_resumes['similarity_score'] >= threshold]
print("\nTop Matching Resumes (Score >= {}):".format(threshold))
print(top_resumes)

# Sort by similarity score in descending order
sorted_resumes = df_resumes.sort_values(by='similarity_score', ascending=False)
print("\nResumes Sorted by Match Score:")
print(sorted_resumes)

# Visualize the results
print("\n" + "="*60)
print("RESUME MATCHING RESULTS")
print("="*60)
for _, row in sorted_resumes.iterrows():
    print(f"\nResume ID: {row['resume_id']}")
    print(f"Score: {row['similarity_score']:.2%}")
    print(f"Text Preview: {row['resume_text'][:80]}...")
    
# Find the best match
best_match = sorted_resumes.iloc[0]
print(f"\n\nBEST MATCH: Resume ID {best_match['resume_id']}")
print(f"Match Score: {best_match['similarity_score']:.2%}")
print(f"Skills: {best_match['resume_text']}")