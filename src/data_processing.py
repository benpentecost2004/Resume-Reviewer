import re
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("AzharAli05/Resume-Screening-Dataset")
df = pd.DataFrame(dataset["train"])

def extract_applicant_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    
    for line in lines[:5]:
        if len(line.split()) <= 5 and not re.search(r'\d', line) and not re.match(r'^(summary|objective|about me)', line.lower()):
            return line.strip().title()

    return lines[0].strip().title() if lines else ""


def extract_sections(resume_text):
    sections = {
        "Summary": "",
        "Experience": "",
        "Skills": "",
        "Education": "",
        "Certifications": "",
        "Projects": ""
    }

    text = re.sub(r'\r', '', resume_text)
    lines = text.splitlines()

    patterns = {
        "Summary": r'(summary|objective|about me)',
        "Experience": r'(experience|employment history|work history)',
        "Skills": r'(skills|technical skills|core competencies)',
        "Education": r'(education|academic background)',
        "Certifications": r'(certifications|licenses)',
        "Projects": r'(projects|personal projects|portfolio)',
    }

    buffer = {k: [] for k in sections.keys()}
    current_section = None

    for line in lines:
        line_lower = line.strip().lower()

        matched_section = None
        for section, pattern in patterns.items():
            if re.match(rf'^\s*{pattern}\b', line_lower):
                matched_section = section
                break

        if matched_section:
            current_section = matched_section
            continue

        if current_section:
            buffer[current_section].append(line.strip())

    for section in sections.keys():
        sections[section] = " ".join(buffer[section]).strip()

    return sections

print("Parsing resumes...")

structured_data = []
for _, row in df.iterrows():
    resume_text = row["Resume"]
    role = row.get("Role", "")
    decision = row.get("Decision", "")

    applicant_name = extract_applicant_name(resume_text)
    sections = extract_sections(resume_text)

    if all(sections[k] == "" for k in sections.keys()):
        continue

    record = {
        "Applicant Name": applicant_name,
        "Role": role,
        "Summary": sections["Summary"],
        "Experience": sections["Experience"],
        "Skills": sections["Skills"],
        "Education": sections["Education"],
        "Certifications": sections["Certifications"],
        "Projects": sections["Projects"],
        "Decision": decision,
    }

    structured_data.append(record)


final_df = pd.DataFrame(structured_data)
final_df = final_df.fillna("")
# final_df.to_csv("dataset.csv", index=False)
final_df.to_json("dataset.json", orient="records", indent=4)

print("Data processing complete!")
print(f"Total resumes processed: {len(final_df)}")
print(final_df.head(5))

print("Columns in final dataset:", final_df.columns.tolist())