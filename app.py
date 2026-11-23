
import streamlit as st
import re
import plotly.express as px
import pandas as pd

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page Configuration
st.set_page_config(
    page_title="SkillBridge AI",
    page_icon="🎓",
    layout="wide"
)


# Professional CSS

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: #222222;
}

h1, h2, h3, p, label {
    color: #222222 !important;
}

.stTextArea textarea {
    background-color: white !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FB;
}

/* All text */
.stApp p,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp label,
.stMarkdown {
    color: #222222 !important;
}

/* Text input */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #222222 !important;
}

/* Input placeholder */
.stTextArea textarea::placeholder {
    color: #777777 !important;
}

</style>
""", unsafe_allow_html=True)



# Skill Database
technical_skills = [
    "Python", "Java", "C++", "C", "SQL",
    "HTML", "CSS", "JavaScript", "React",
    "Django", "Machine Learning",
    "Deep Learning", "Artificial Intelligence",
    "Data Science", "Pandas", "NumPy",
    "Scikit-learn", "TensorFlow", "Flask",
    "Git", "GitHub", "Data Structures",
    "Algorithms"
]


# Learning Resources
learning_resources = {
    "Python": "Learn Python fundamentals",
    "Java": "Learn Java and OOP concepts",
    "C++": "Learn C++ programming",
    "SQL": "Practice SQL queries",
    "Django": "Learn Django web development",
    "HTML": "Learn HTML fundamentals",
    "CSS": "Practice CSS styling",
    "JavaScript": "Learn JavaScript",
    "Machine Learning": "Study ML algorithms",
    "Deep Learning": "Study neural networks",
    "Data Structures": "Practice arrays, stacks and queues",
    "Algorithms": "Practice sorting and searching",
    "Git": "Learn Git version control",
    "GitHub": "Practice GitHub"
}


# Extract Skills
def extract_skills(text):

    extracted = []

    text = text.lower()

    for skill in technical_skills:

        pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'

        if re.search(pattern, text):
            extracted.append(skill)

    return extracted


# Compare Skills
def compare_skills(resume_skills, required_skills):

    matching = [
        skill for skill in required_skills
        if skill in resume_skills
    ]

    missing = [
        skill for skill in required_skills
        if skill not in resume_skills
    ]

    extra = [
        skill for skill in resume_skills
        if skill not in required_skills
    ]

    percentage = (
        len(matching) / len(required_skills) * 100
        if required_skills else 0
    )

    return matching, missing, extra, percentage


# Calculate Text Similarity
def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return similarity[0][0] * 100


# Read Resume PDF
def read_resume(uploaded_file):

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:
        resume_text += page.extract_text() or ""

    return resume_text


# Analyze Resume
def analyze_resume(uploaded_file, job_description):

    resume_text = read_resume(uploaded_file)

    if not resume_text.strip():
        return None

    resume_skills = extract_skills(resume_text)

    required_skills = extract_skills(job_description)

    matching, missing, extra, percentage = compare_skills(
        resume_skills,
        required_skills
    )

    try:
        similarity = calculate_similarity(
            resume_text,
            job_description
        )
    except ValueError:
        similarity = None

    recommendations = [
        learning_resources.get(
            skill,
            "Find learning resources for " + skill
        )
        for skill in missing
    ]

    return {
        "matching": matching,
        "missing": missing,
        "extra": extra,
        "percentage": percentage,
        "similarity": similarity,
        "recommendations": recommendations
    }


# Application Title
st.title("🎓 SkillBridge AI")

st.subheader("AI-Powered Resume and Job Skill Analyzer")

st.write(
    "Analyze resumes, identify skill gaps "
    "and discover career opportunities."
)

st.divider()


# Select User Mode
st.sidebar.title("Dashboard")

user_mode = st.sidebar.radio(
    "Select Your Role",
    ["Candidate Mode", "HR Mode"]
)


# Job Description
st.header("Job Requirements")

job_description = st.text_area(
    "Enter Job Description",
    height=180,
    placeholder="Enter required skills and job details..."
)


# ==================================================
# CANDIDATE DASHBOARD
# ==================================================

if user_mode == "Candidate Mode":

    st.header("👨‍🎓 Candidate Dashboard")

    uploaded_file = st.file_uploader(
        "Upload Your Resume",
        type=["pdf"],
        key="candidate_resume"
    )

    if st.button("Analyze My Skills", type="primary"):

        if uploaded_file is None:
            st.warning("Please upload your resume.")

        elif not job_description.strip():
            st.warning("Please enter a job description.")

        else:

            try:

                result = analyze_resume(
                    uploaded_file,
                    job_description
                )

                if result is None:

                    st.error("No readable text found in this PDF.")

                else:

                    matching = result["matching"]
                    missing = result["missing"]
                    extra = result["extra"]
                    percentage = result["percentage"]
                    similarity = result["similarity"]
                    recommendations = result["recommendations"]

                    st.divider()

                    st.header("📊 Candidate Analysis Results")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "🎯 Skill Match",
                        f"{percentage:.2f}%"
                    )

                    col2.metric(
                        "Text Similarity",
                        f"{similarity:.2f}%"
                        if similarity is not None
                        else "Unavailable"
                    )

                    col3.metric(
                        "Matching Skills",
                        len(matching)
                    )

                    col4.metric(
                        "Missing Skills",
                        len(missing)
                    )

                    # Progress
                    st.subheader("Skill Match Progress")

                    st.progress(int(percentage))

                    # Circular Chart
                    st.subheader("🎯 Overall Skill Match")

                    gauge_data = pd.DataFrame({
                        "Status": ["Matched", "Remaining"],
                        "Percentage": [
                            percentage,
                            100 - percentage
                        ]
                    })

                    gauge_fig = px.pie(
                        gauge_data,
                        names="Status",
                        values="Percentage",
                        hole=0.7,
                        color="Status",
                        color_discrete_map={
                            "Matched": "#2563EB",
                            "Remaining": "#E5E7EB"
                        }
                    )

                    gauge_fig.update_traces(textinfo="none")

                    gauge_fig.update_layout(
                        annotations=[{
                            "text": f"{percentage:.1f}%",
                            "x": 0.5,
                            "y": 0.5,
                            "font_size": 28,
                            "showarrow": False
                        }],
                        height=350
                    )

                    st.plotly_chart(
                        gauge_fig,
                        use_container_width=True
                    )

                    # Skill Details
                    st.subheader("Matching Skills")

                    st.success(
                        ", ".join(matching)
                        if matching else "No matching skills found."
                    )

                    st.subheader("Missing Skills")

                    st.warning(
                        ", ".join(missing)
                        if missing else "No missing skills!"
                    )

                    st.subheader("Additional Skills")

                    st.info(
                        ", ".join(extra)
                        if extra else "No additional skills found."
                    )

                    # Bar Chart
                    st.subheader("📊 Skill Analysis")

                    chart_data = pd.DataFrame({
                        "Category": [
                            "Matching Skills",
                            "Missing Skills",
                            "Extra Skills"
                        ],
                        "Count": [
                            len(matching),
                            len(missing),
                            len(extra)
                        ]
                    })

                    fig = px.bar(
                        chart_data,
                        x="Category",
                        y="Count",
                        title="Skill Gap Analysis",
                        text="Count"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # Pie Chart
                    st.subheader("🥧 Skill Distribution")

                    pie_fig = px.pie(
                        chart_data,
                        names="Category",
                        values="Count",
                        title="Overall Skill Distribution",
                        hole=0.4
                    )

                    st.plotly_chart(
                        pie_fig,
                        use_container_width=True
                    )

                    # Recommendations
                    st.header("4. Learning Recommendations")

                    if recommendations:

                        for recommendation in recommendations:
                            st.info("📚 " + recommendation)

                    else:
                        st.success(
                            "You have all identified required skills!"
                        )

                    # Download Report
                    report_text = f"""
SKILLBRIDGE AI - CANDIDATE REPORT

Skill Match: {percentage:.2f}%

Text Similarity: {
    f"{similarity:.2f}%"
    if similarity is not None
    else "Unavailable"
}

Matching Skills:
{", ".join(matching)}

Missing Skills:
{", ".join(missing)}

Additional Skills:
{", ".join(extra)}

Learning Recommendations:
{chr(10).join(recommendations)}
"""

                    st.header("5. Download Your Report")

                    st.download_button(
                        "📥 Download Analysis Report",
                        data=report_text,
                        file_name="SkillBridge_Candidate_Report.txt",
                        mime="text/plain"
                    )

                    st.success("Resume analysis completed!")

            except Exception as e:
                st.error(f"An error occurred: {e}")


# ==================================================
# HR DASHBOARD
# ==================================================

else:

    st.header("🏢 HR Recruitment Dashboard")

    st.write(
        "Upload multiple candidate resumes "
        "to analyze and compare their skills."
    )

    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (Multiple PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        key="hr_resumes"
    )

    if uploaded_files:

        st.info(
            f"{len(uploaded_files)} candidate resume(s) uploaded."
        )

    if st.button("Analyze All Candidates", type="primary"):

        if not uploaded_files:

            st.warning("Please upload candidate resumes.")

        elif not job_description.strip():

            st.warning("Please enter job requirements.")

        else:

            results = []

            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):

                try:

                    result = analyze_resume(
                        uploaded_file,
                        job_description
                    )

                    if result is None:
                        st.warning(
                            f"Could not read {uploaded_file.name}"
                        )
                        continue

                    # Candidate name from filename
                    candidate_name = re.sub(
                        r'\.pdf$',
                        '',
                        uploaded_file.name,
                        flags=re.IGNORECASE
                    )

                    results.append({
                        "Candidate": candidate_name,
                        "Skill Match": result["percentage"],
                        "Text Similarity": result["similarity"],
                        "Matching Skills": len(result["matching"]),
                        "Missing Skills": len(result["missing"]),
                        "Missing Skill List": ", ".join(result["missing"]),
                        "Matching Skill List": ", ".join(result["matching"])
                    })

                except Exception as e:

                    st.warning(
                        f"Error analyzing {uploaded_file.name}: {e}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            if results:

                # Sort candidates
                results = sorted(
                    results,
                    key=lambda x: x["Skill Match"],
                    reverse=True
                )

                for rank, candidate in enumerate(results, start=1):
                    candidate["Rank"] = rank

                df = pd.DataFrame(results)

                st.divider()

                st.header("📊 HR Screening Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Candidates",
                    len(results)
                )

                col2.metric(
                    "Average Skill Match",
                    f"{df['Skill Match'].mean():.2f}%"
                )

                col3.metric(
                    "Highest Skill Match",
                    f"{df['Skill Match'].max():.2f}%"
                )

                # Candidate Comparison
                st.subheader("👥 Candidate Comparison")

                display_df = df[
                    [
                        "Rank",
                        "Candidate",
                        "Skill Match",
                        "Text Similarity",
                        "Matching Skills",
                        "Missing Skills"
                    ]
                ].copy()

                display_df["Skill Match"] = (
                    display_df["Skill Match"].map(
                        lambda x: f"{x:.2f}%"
                    )
                )

                display_df["Text Similarity"] = (
                    display_df["Text Similarity"].map(
                        lambda x: f"{x:.2f}%"
                        if pd.notna(x)
                        else "Unavailable"
                    )
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Ranking Chart
                st.subheader("🏆 Candidate Skill Match Ranking")

                ranking_fig = px.bar(
                    df,
                    x="Candidate",
                    y="Skill Match",
                    color="Skill Match",
                    text="Skill Match",
                    title="Candidate Skill Match Comparison"
                )

                ranking_fig.update_layout(
                    yaxis_title="Skill Match Percentage",
                    xaxis_title="Candidate"
                )

                st.plotly_chart(
                    ranking_fig,
                    use_container_width=True
                )

                # Individual Candidate Details
                st.header("Candidate Screening Details")

                for candidate in results:

                    with st.expander(
                        f"Rank {candidate['Rank']} - "
                        f"{candidate['Candidate']}"
                    ):

                        st.metric(
                            "Skill Match",
                            f"{candidate['Skill Match']:.2f}%"
                        )

                        st.write(
                            "**Matching Skills:**",
                            candidate["Matching Skill List"] or "None"
                        )

                        st.write(
                            "**Missing Skills:**",
                            candidate["Missing Skill List"] or "None"
                        )

                # Download HR Report
                st.header("📥 Download HR Screening Report")

                download_df = df[
                    [
                        "Rank",
                        "Candidate",
                        "Skill Match",
                        "Text Similarity",
                        "Matching Skills",
                        "Missing Skills",
                        "Matching Skill List",
                        "Missing Skill List"
                    ]
                ]

                csv_report = download_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download HR Screening CSV",
                    data=csv_report,
                    file_name="SkillBridge_HR_Screening_Report.csv",
                    mime="text/csv"
                )

                st.success(
                    "All readable candidate resumes analyzed successfully!"
                )

                st.caption(
                    "Ranking is based only on extracted skill matches. "
                    "It is an initial screening aid, not a final hiring decision."
                )

            else:

                st.error(
                    "No readable candidate resumes were found."
                )
