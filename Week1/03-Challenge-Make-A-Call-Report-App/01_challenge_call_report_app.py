# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# CHALLENGE #1 - CREATE A CALL REPORT APP
# ***

# Challenge Instructions:
#  1. Use this prompt template to develop a call report from the Nike Earnings Call Transcripts
#  2. Remove the toggle button for the how the results are summarized


prompt_template = """
Write a business report from the following earnings call transcript:
{text}

Use the following Markdown format:
# Insert Descriptive Report Title

## Earnings Call Summary
Use 3 to 7 numbered bullet points

## Important Financials
Describe the most important financials discussed during the call. Use 3 to 5 numbered bullet points.

## Key Business Risks
Describe any key business risks discussed on the call. Use 3 to 5 numbered bullets.

## Conclusions
Conclude with any overaching business actions that the company is pursuing that may have a positive or negative implications and what those implications are. 
"""
