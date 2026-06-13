Machine Learning-Based CV Screening and Candidate Ranking System for Public Service Recruitment
==============================================================================================

Project Overview
----------------
This project was developed as part of the BTEC Unit 2 Independent Project and final-year graduation project.

The system uses machine learning to analyse candidate CVs, predict suitable IT roles, compare CVs with job descriptions, and rank applicants based on how well they match a selected vacancy.

The project is designed as an AI-assisted recruitment prototype for public service organisations. It helps HR specialists reduce manual CV screening time, improve consistency, and view clear candidate recommendations.

The system does not make final hiring decisions. Final decisions should always be reviewed and approved by human HR specialists.

Project Aim
-----------
The aim of this project is to design and implement a machine learning-based CV screening and candidate ranking system that helps HR specialists review applicants more efficiently, consistently, and transparently in public service recruitment.

Project Objectives
------------------
The main objectives are:

1. To investigate the challenges of manual CV screening in public service recruitment.
2. To prepare and use an anonymized IT resume dataset for machine learning model training.
3. To implement a supervised machine learning model for CV role classification.
4. To compare candidate CVs with job descriptions using text similarity and skill matching.
5. To calculate candidate suitability scores and generate ranked results.
6. To display Top 3 recommended candidates for HR review.
7. To provide understandable explanations for candidate recommendations.
8. To consider ethical issues such as privacy, bias, transparency, and human oversight.

Main Features
-------------
The system includes:

- Upload and analyse multiple candidate CVs.
- Support PDF, DOCX, and TXT CV file formats.
- Extract text from uploaded CV files.
- Clean and preprocess CV text.
- Predict the most suitable IT role using a trained machine learning model.
- Compare CV content with a selected job description.
- Identify matched and missing skills.
- Calculate final candidate suitability scores.
- Apply role matching against the selected target role.
- Rank candidates based on final score.
- Display Top 3 recommended candidates only when they match the selected role and meet the minimum score.
- Show all analysed candidates in a ranking table.
- Provide OpenAI-assisted or rule-based explanations.
- Mask sensitive information such as emails, phone numbers, links, and address-like text.
- Keep final hiring decisions under human HR review.

Technologies Used
-----------------
Programming language:

- Python

Frameworks and libraries:

- Streamlit
- pandas
- NumPy
- scikit-learn
- joblib
- pdfplumber
- python-docx
- python-dotenv
- OpenAI API, optional
- Regular Expressions, regex

Machine learning tools:

- TF-IDF Vectorization
- Logistic Regression
- Stratified train/test split
- Classification report
- Confusion matrix
- Accuracy, precision, recall, and F1-score

Storage / database:

- CSV-based storage is used for the dataset and job description templates.
- No external database is required for the current prototype.

Dataset Information
-------------------
The project uses a synthetic anonymized IT resume dataset created for academic and experimental purposes.

The dataset includes candidate profiles for these IT roles:

- Frontend Developer
- Backend Developer
- AI Engineer
- Machine Learning Engineer
- QA Engineer
- Flutter Developer

The dataset includes these seniority levels:

- Junior
- Middle
- Senior

The dataset does not contain real names, phone numbers, emails, addresses, or private personal information. It was created to reduce privacy risks and make the project suitable for academic testing.

Dataset Files
-------------
The dataset files are stored in the data/ folder:

```text
data/
|-- it_cv_synthetic_dataset_180.csv
|-- it_cv_synthetic_dataset_900.csv
|-- it_cv_labels.csv
`-- it_job_description_templates_18.csv
```

For the final model training, the system uses:

```text
data/it_cv_synthetic_dataset_900.csv
```

Dataset Version Explanation
---------------------------
This project includes two dataset files because the dataset was improved during development.

The first dataset, `it_cv_synthetic_dataset_180.csv`, was created during the initial prototype stage. It contains 180 synthetic anonymized IT resumes. It was useful for testing the first version of the machine learning model, CV text processing, role prediction, and candidate ranking logic.

After initial testing, the 180-row dataset was found to be too limited for realistic candidate ranking. There were fewer examples per role and seniority level. Some roles also had overlapping skills, which made the Top 3 ranking less reliable.

To improve the project, an expanded dataset was created.

The improved dataset, `it_cv_synthetic_dataset_900.csv`, contains 900 synthetic anonymized resumes across the same IT roles and seniority levels. It includes more variation in skills, experience, tools, responsibilities, certifications, and public service project contexts.

The 900-row dataset includes:

- 150 resumes per role.
- 50 Junior resumes per role.
- 50 Middle resumes per role.
- 50 Senior resumes per role.
- 900 resumes in total.

Both datasets are kept in the project folder for documentation and comparison. The 180-row dataset shows the initial development stage. The 900-row dataset shows the improved dataset used for the final training stage.

Keeping both files demonstrates project iteration. It also shows how the dataset was improved to increase model reliability and candidate ranking accuracy.

Machine Learning Approach
-------------------------
This project uses a supervised machine learning approach.

The system learns from labelled resume data. Each CV is connected to a target IT role. The trained model then predicts which role a new candidate CV is most suitable for.

TF-IDF Vectorization
--------------------
TF-IDF Vectorization is used to convert CV text into numerical features that the machine learning model can understand.

In simple terms, TF-IDF helps the system identify important words and skills in a CV, such as:

- Python
- React
- SQL
- Selenium
- Flutter
- Machine Learning
- REST API
- Docker

The text is converted into numerical values so that the model can process it.

Logistic Regression
-------------------
Logistic Regression is used as the supervised classification model.

In this project, Logistic Regression predicts the most suitable IT role for each CV based on the text features created by TF-IDF.

Example:

```text
CV skills: Postman, Jira, API Testing, SQL, Selenium
Predicted role: QA Engineer
```

Candidate Scoring Method
------------------------
The final candidate score is calculated using three components:

```text
Final Score = 50% AI model confidence
            + 30% job description similarity
            + 20% skill match score
```

Score components:

1. AI Model Confidence
   Shows how confident the machine learning model is about the predicted role.

2. Job Description Similarity
   Measures how closely the candidate CV matches the job description.

3. Skill Match Score
   Compares required skills from the job description with the skills found in the CV.

Role Match Logic
----------------
The user selects the target role in the Streamlit app.

The system compares:

```text
predicted_role == selected_target_role
```

If the predicted role matches the selected target role:

```text
Role Match = Yes
```

If the predicted role does not match the selected target role:

```text
Role Match = No
```

If Role Match is No, the final score is reduced by 50%.

This prevents irrelevant candidates from appearing in the Top 3 recommendations. For example, a Flutter Developer CV should not be recommended for a Backend Developer vacancy if the role does not match.

Recommendation Levels
---------------------
The system uses these recommendation rules:

```text
80-100 and Role Match = Yes: Strong Fit
60-79 and Role Match = Yes: Medium Fit
Below 60 or Role Match = No: Weak Fit / Not Recommended
```

Top 3 recommended candidates only include candidates where:

```text
Role Match = Yes
Final Score >= 60
```

If no candidate meets these conditions, the app shows:

```text
No suitable candidates found for this role. Please review more CVs or adjust the job description.
```

Project Structure
-----------------
```text
CV_Screening_System/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.txt
|-- User_Manual.md
|-- data/
|   |-- it_cv_synthetic_dataset_180.csv
|   |-- it_cv_synthetic_dataset_900.csv
|   |-- it_cv_labels.csv
|   `-- it_job_description_templates_18.csv
|-- models/
|   |-- resume_classifier.pkl
|   `-- tfidf_vectorizer.pkl
|-- outputs/
|   |-- evaluation_report.txt
|   `-- predictions_sample.csv
`-- utils/
    |-- __init__.py
    |-- text_extraction.py
    |-- preprocessing.py
    |-- skills.py
    |-- scoring.py
    `-- openai_explanation.py
```

Project Architecture
--------------------
This project is a Streamlit machine learning application.

It does not use a separate frontend, backend, and database server. Instead, Streamlit allows the user interface and Python logic to work together in one app.

In this project:

- The frontend is implemented in `app.py`.
  Users select a target role, enter a job description, upload CV files, and view ranked results.
- The backend logic is written in Python.
  It extracts CV text, cleans text, predicts roles, matches skills, calculates scores, and creates explanations.
- The database layer is simple CSV file storage.
  The data folder stores the synthetic CV datasets, labels, and job description templates.

This structure is suitable for a university prototype. It is easy to run locally and clearly shows the machine learning workflow.

Installation Guide
------------------
1. Open the project folder:

```bash
cd CV_Screening_System
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment.

For macOS:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

4. Install required packages:

```bash
python -m pip install -r requirements.txt
```

How to Train the Model
----------------------
Before running the application, train the machine learning model:

```bash
python train_model.py
```

This command will:

- Load the resume dataset.
- Preprocess the CV text.
- Split the data into training and testing sets.
- Train the TF-IDF + Logistic Regression model.
- Evaluate the model.
- Save the trained model and vectorizer into the models/ folder.
- Generate an evaluation report in the outputs/ folder.

Expected output files:

```text
models/
|-- resume_classifier.pkl
`-- tfidf_vectorizer.pkl

outputs/
|-- evaluation_report.txt
`-- predictions_sample.csv
```

How to Run the Application
--------------------------
After training the model, run the Streamlit app:

```bash
streamlit run app.py
```

If the command does not work, use:

```bash
python -m streamlit run app.py
```

How to Use the Platform
-----------------------
1. Open the Streamlit application.
2. Select the target role for the vacancy.
3. Paste the job description.
4. Upload one or more candidate CV files.
5. Click **Start CV Analysis**.
6. Review the Candidate Ranking Results table.
7. Check the Top 3 Recommended Candidates section.
8. Open candidate details to view:
   - selected target role
   - predicted role
   - role match status
   - AI confidence
   - job match score
   - skill match score
   - role mismatch penalty
   - final score
   - recommendation
   - matched skills
   - missing skills
   - explanation
   - HR review note

Optional OpenAI API Configuration
---------------------------------
The system can optionally use the OpenAI API to generate more natural candidate explanations.

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Important:

- Do not write the API key directly inside Python files.
- Do not upload the `.env` file to GitHub.
- If no API key is provided, the system still works using rule-based explanations.
- If the OpenAI API is unavailable, the app falls back to rule-based explanations.

Ethical Considerations
----------------------
This project considers the following ethical issues.

Data privacy:

- The dataset is synthetic and anonymized.
- It does not include real candidate names, emails, phone numbers, addresses, or private personal information.
- Emails, phone numbers, links, and address-like text are masked during preprocessing.

Human oversight:

- The system supports HR specialists but does not replace them.
- Final hiring decisions must always be made by human reviewers.

Bias risk:

- AI recruitment systems can produce biased results if the dataset is biased or incomplete.
- This project uses a controlled synthetic dataset and includes explanations to improve transparency.

Transparency:

- The system shows matched skills, missing skills, final scores, role match status, and explanations.
- This helps HR users understand why a candidate was recommended or not recommended.

BTEC Report Guidance
--------------------
Suggested written report sections:

1. Project Proposal
   Explain the recruitment problem, project aim, intended users, and why machine learning is suitable.

2. Planning
   Include objectives, timeline, resources, risks, and success criteria.

3. Methodology
   Describe the dataset, preprocessing, TF-IDF feature extraction, Logistic Regression model, ranking method, and evaluation method.

4. Implementation
   Explain how train_model.py, app.py, and utility modules work.

5. Findings and Analysis
   Use outputs/evaluation_report.txt to discuss model performance.

6. Evaluation
   Evaluate accuracy, precision, recall, F1-score, usability, explainability, and limitations.

7. Conclusion
   Summarize whether the system met the project objectives.

8. Reflection
   Discuss what went well, what was difficult, and how the system could be improved.

Limitations
-----------
The current system has several limitations:

- The dataset is synthetic and may not fully represent real-world CV complexity.
- Real CVs may contain different formats, missing sections, spelling errors, or unclear descriptions.
- The model may confuse roles with overlapping skills, such as AI Engineer and Machine Learning Engineer.
- The system cannot evaluate soft skills, personality, motivation, or interview performance.
- The system provides recommendations only and should not be used as the sole basis for hiring decisions.

Future Improvements
-------------------
Future improvements may include:

- Using a larger anonymized real-world CV dataset collected with consent.
- Improving role detection and job description understanding.
- Adding better seniority-level prediction.
- Adding bias detection and fairness analysis.
- Improving explainability with more detailed evidence.
- Adding database storage.
- Adding user authentication.
- Creating an admin dashboard for HR specialists.
- Integrating the system with real recruitment platforms.

Academic Context
----------------
This project was developed for:

```text
BTEC Unit 2 Independent Project
Final-Year Graduation Project
Digital Technologies / Artificial Intelligence Specialism
```

The project demonstrates practical application of machine learning, natural language processing, data analysis, ethical awareness, and professional software development in the context of public service recruitment.

Author
------
Student: Munisa Tulanova

Project Theme: Machine Learning-Based Automatic Screening System for Job Applicants' CVs in Public Service

Project Type: Practice-based / Capstone-style Independent Project
