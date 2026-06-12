# User Manual

## System Name

Machine Learning-Based Automatic CV Screening and Candidate Ranking System

## Intended User

This system is designed for a recruiter, HR assistant, public service recruitment officer, or student evaluator who wants to screen IT CVs against a job description.

## What the System Does

The system:

- Trains a supervised machine learning model using labelled synthetic CV data.
- Predicts a candidate's most likely IT role.
- Compares each CV with a pasted job description.
- Extracts matched and missing skills.
- Calculates a final candidate score.
- Ranks candidates and highlights the Top 3.
- Provides an explanation for each top-ranked candidate.

## Before You Start

Make sure these dataset files are inside the `data/` folder:

- `it_cv_synthetic_dataset_180.csv`
- `it_cv_labels.csv`
- `it_job_description_templates_18.csv`

Install the required packages:

```bash
pip install -r requirements.txt
```

## Step 1: Train the Model

Run:

```bash
python train_model.py
```

Expected result:

- `models/resume_classifier.pkl` is created.
- `models/tfidf_vectorizer.pkl` is created.
- `outputs/evaluation_report.txt` is created.
- `outputs/predictions_sample.csv` is created.

If the dataset is missing, the script will show a clear error message telling you which file is needed.

## Step 2: Start the Web App

Run:

```bash
streamlit run app.py
```

Streamlit will open the app in your web browser.

## Step 3: Paste the Job Description

Paste the vacancy or role description into the job description box.

For best results, include:

- Job title
- Required programming languages
- Required frameworks
- Required tools
- Experience expectations
- Key responsibilities

## Step 4: Upload CV Files

Upload one or more candidate CV files.

Supported formats:

- PDF
- DOCX
- TXT

## Step 5: Screen and Rank Candidates

Click:

```text
Screen and Rank Candidates
```

The app will show:

- Candidate rank
- File name
- Predicted role
- ML confidence
- Job description similarity
- Skill match score
- Final score
- Recommendation
- Matched skills
- Missing skills

## Final Score Formula

The system ranks candidates using:

```text
Final Score = 50% ML confidence + 30% job description similarity + 20% skill match score
```

Recommendation rules:

- 80-100 = Strong Fit
- 60-79 = Medium Fit
- Below 60 = Weak Fit

## Step 6: Review the Top 3

The Top 3 section provides more detail:

- Matched skills
- Missing skills
- Explanation of the result

The explanation is generated using OpenAI if an API key is available. If no key is available, the system uses a built-in rule-based explanation. The explanation is only a recommendation for HR review and must not be used as a final hiring decision.

## Step 7: Download Results

Click:

```text
Download ranking as CSV
```

This exports the ranking table for reporting or evidence.

## Ethical Use Guidance

This system is a decision-support tool. It should not make final hiring decisions by itself.

Good practice:

- Use anonymized CVs where possible.
- Do not include age, gender, ethnicity, disability, religion, or other protected characteristics in scoring.
- Use human review for final decisions.
- Keep candidate data private and secure.
- Check whether the model performs fairly across different candidate groups.
- Emails, phone numbers, URLs, and address-like text are masked during preprocessing.
- Do not send raw CV text or candidate personal details to explanation tools.
- OpenAI explanations receive only job-related scores and skill evidence.

## Troubleshooting

### The app says model files are missing

Run:

```bash
python train_model.py
```

### Training says dataset files are missing

Check that the CSV files are inside the `data/` folder.

### A PDF CV cannot be read

Some scanned PDFs contain images instead of selectable text. Convert the CV using OCR or upload a DOCX/TXT version.

### OpenAI explanation does not work

The app will still work. It will use a rule-based explanation.

To enable OpenAI explanations, create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

## Suggested Evidence for BTEC Submission

Include screenshots of:

- Dataset files in the folder
- Training script running
- Evaluation report
- Streamlit app home screen
- Job description input
- CV upload
- Ranked results table
- Top 3 candidate explanations
