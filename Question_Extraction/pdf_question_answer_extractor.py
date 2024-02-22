
# we begin the extraction for: "American College of Physicians -  MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021)"
import fitz
import pandas as pd
import re
import glob

# get file
pdf_path = 'Questions/American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).pdf'

doc = fitz.open(pdf_path)

# Initialize an empty list to store questions and answers
questions_list = []
answers_list = []
# Current question and answer placeholders
current_question = []
current_answer = []

# Regular expressions to identify questions and answers
question_regex = r'^Item \d'
question_regex_variety = r'llem \d'
answer_regex = r'^Answer:'
answer_regex_variety = r'llem\d '

# Encounter question, answer
found_question = False
found_answer = False
found_stop_word = False

stop_word = 'Bibliography'
# Extract text from each page
# questions start at page 82
for page_num in range(82, len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()

    # Split text into lines for processing
    lines = text.split('\n')

    for line in lines:
        # Check if the line is a question
        if re.match(question_regex, line) or re.match(question_regex_variety, line):
            # If there's a current question being processed, save it before moving on to the next
            found_question = True
            found_stop_word = False
            # if we continue to next question, add current one to the list
            if current_question:
                questions_list.append(' '.join(current_question))
            current_question = []

        # check if it's a answer
        elif re.match(answer_regex, line) or re.match(answer_regex_variety, line) or found_stop_word:
            found_answer = True if not found_stop_word else False
            if current_answer:
                answers_list.append(' '.join(current_answer))
            current_answer = [line] if re.match(answer_regex, line) else []
            found_question = False
            found_stop_word = False

        # # a stop_word was found
        # elif found_stop_word:
        #     answers_list.append(' '.join(current_answer))
        #     current_answer = []
        #     found_stop_word = False
        elif stop_word in line:
            found_stop_word = True
            continue

        elif found_question and not found_stop_word:
            current_question.append(line)

        elif found_answer and not found_stop_word:
            current_answer.append(line)


# Convert the list of Q&A pairs into a DataFrame
hematology_qa = pd.DataFrame({'Questions': questions_list, 'Answers': answers_list})

# Save the DataFrame to a CSV file
csv_path = 'Questions/American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).csv'
hematology_qa.to_csv(csv_path, index=False)

print(f'Successfully extracted Q&A to {csv_path}')
