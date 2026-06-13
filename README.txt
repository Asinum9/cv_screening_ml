Machine Learning-Based Automatic CV Screening and Candidate Ranking System
=========================================================================

Project Title
-------------
Machine Learning-Based Automatic CV Screening and Candidate Ranking System for Public Service Recruitment

Purpose
-------
This beginner-friendly Python project demonstrates a supervised machine learning system that screens CVs, predicts candidate role suitability, compares CVs with a pasted job description, ranks candidates, and explains the ranking result.

The project is suitable for a BTEC Unit 2 Independent Project because it includes:
- Project proposal and planning support
- Data preprocessing
- Machine learning model training
- Prediction and ranking
- Evaluation metrics
- Findings and analysis outputs
- Ethical and privacy handling
- Explainable results
- Reflection points for improvement

Project Structure
-----------------
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

Project Architecture
--------------------
This project is a Streamlit machine learning application.

It does not use a separate frontend, backend, and database server.
Instead, Streamlit allows the user interface and Python logic to work together in one app.

In this project:
- The frontend is in app.py.
  This is where the user enters a job description, uploads CV files, and views the ranked results.
- The backend logic is written in Python.
  It extracts CV text, cleans the text, predicts the candidate role, matches skills, calculates scores, and creates explanations.
- The database layer is simple CSV file storage.
  The data folder stores the synthetic CV dataset, labels, and job description templates.

This structure is suitable for a university prototype.
It is easy to run on one computer.
It also clearly shows the machine learning workflow needed for the project.

Important Note About Dataset Files
----------------------------------
Place these prepared files inside the data/ folder before training:
- data/it_cv_synthetic_dataset_900.csv
- data/it_cv_labels.csv
- data/it_job_description_templates_18.csv

The dataset was expanded from 180 to 900 synthetic anonymized resumes.
This improves model reliability and candidate ranking accuracy.
The new dataset includes 150 resumes per role.
Each role has 50 Junior, 50 Middle, and 50 Senior resumes.

The model files in models/ are created after running train_model.py.
The evaluation files in outputs/ are also created after training.

Installation
------------
1. Open a terminal in the CV_Screening_System folder.
2. Create a virtual environment:

   python -m venv .venv

3. Activate it:

   macOS/Linux:
   source .venv/bin/activate

   Windows:
   .venv\Scripts\activate

4. Install dependencies:

   pip install -r requirements.txt

Training the Model
------------------
After placing the CSV files in data/, run:

   python train_model.py

This will:
- Load the dataset
- Clean the CV text
- Train a TF-IDF + Logistic Regression classifier
- Evaluate accuracy, precision, recall, F1-score, and confusion matrix
- Save the model to models/resume_classifier.pkl
- Save the vectorizer to models/tfidf_vectorizer.pkl
- Save outputs/evaluation_report.txt
- Save outputs/predictions_sample.csv

Running the App
---------------
Run:

   streamlit run app.py

Then:
1. Select the target role.
2. Paste a job description.
3. Upload multiple CV files in PDF, DOCX, or TXT format.
4. Click "Start CV Analysis".
5. Review predicted role, role match, ML confidence, job description similarity, skill match score, final score, recommendation, matched skills, missing skills, ranking, and explanation.

Machine Learning Method
-----------------------
This project uses:
- TF-IDF vectorization to convert CV text into numerical features
- Logistic Regression for supervised role classification
- Cosine similarity to compare CV text with the job description
- Skill matching to identify matched and missing job-related skills
- Weighted scoring to rank candidates
- Role matching to compare the predicted CV role with the selected target role

Final Score Formula
-------------------
Final score is calculated as:
- 50% ML confidence
- 30% job description similarity
- 20% skill match score

Role match adjustment:
- The user selects the target role in the app.
- If the predicted CV role matches the selected target role, the score is kept.
- If the predicted CV role does not match the selected target role, the final score is reduced by 50%.
- Top 3 recommended candidates only include candidates with Role Match = Yes and Final Score >= 60.

Recommendation rules:
- 80-100 and Role Match = Yes = Strong Fit
- 60-79 and Role Match = Yes = Medium Fit
- Below 60 or Role Match = No = Weak Fit / Not Recommended

Ethical and Privacy Handling
----------------------------
This project uses synthetic anonymized CV data. In a real recruitment setting:
- CVs should be anonymized where possible.
- Protected characteristics must not be used for scoring.
- The model should support human decision-making, not replace it.
- The system should be tested for fairness across different candidate groups.
- Candidate data should be stored securely and deleted when no longer needed.
- Emails, phone numbers, URLs, and address-like text are masked during preprocessing.
- OpenAI explanations do not receive raw CV text, candidate names, phone numbers, emails, addresses, or uploaded file names.
- Explanations must be treated as recommendations for HR review, not final hiring decisions.

OpenAI Explanation
------------------
The system can generate a short HR-friendly explanation using the OpenAI API if an API key is available.

Create a .env file:

   OPENAI_API_KEY=your_api_key_here

If no API key is available, the app automatically shows a rule-based explanation.
The explanation is only a recommendation for HR review and must not make final hiring decisions.

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
- Synthetic data may not fully represent real CVs.
- Logistic Regression is interpretable and beginner-friendly but less powerful than large language models.
- Skill matching uses a defined skill bank and may miss unusual wording.
- The system should not be used as the only decision-maker in real recruitment.
