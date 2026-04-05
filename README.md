# Volks Legal AI — Multilingual Legal Triage

## Project Overview
A real-world simulation for legal aid. The environment processes complaints in **English, Kannada, and Hindi**, helping users like Ramesh (land disputes) or Rahul (salary issues) navigate the legal system.

## Action Space
1. `classify_case`: Categorize the issue.
2. `set_priority`: Determine legal urgency.
3. `generate_guidance`: 3-step advice in the user's native tongue.
4. `assign_lawyer`: Match based on category and city (Bengaluru, Mumbai, Delhi).

## Rewards
- **+0.25**: Classification
- **+0.20**: Priority
- **+0.25**: Local Language Guidance
- **+0.30**: Correct Lawyer Mapping

## Deployment
1. Create HF Space (Docker).
2. Set `HF_TOKEN`.
3. Tag with `openenv`.