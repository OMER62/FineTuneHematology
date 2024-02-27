
# we begin the extraction for: "American College of Physicians -  MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021)"
import fitz
import pandas as pd
import re
import glob

second_questionnaire_pdf_path = 'Questions/Harrisons Hematology and Oncology 2ed_questionsPage743.pdf'
second_questionnaire_start_page = 743
#second_questionnaire_start_page = 802
second_questionnaire_question_regex = r'^[\s]*\d*\. '
second_questionnaire_question_variety = '(Continued)'
second_questionnaire_answer_id = 'The answer is'
second_questionnaire_stop_word = 'The answers are'
second_questionnaire_csv_path = 'Questions/Harrisons Hematology and Oncology 2ed_questionsPage743.csv'
def extract_qa_from_questionnaire(pdf_path, start_page, question_regex, question_variety, answer, stop_word, csv_path):

    doc = fitz.open(pdf_path)

    # Initialize an empty list to store questions and answers
    questions_list = []
    answers_list = []
    # Current question and answer placeholders
    current_question = []
    current_answer = []



    # Encounter question, answer
    found_question = False
    found_answer = False
    found_stop_word = False

# Extract text from each page
# questions start at page 82
    for page_num in range(start_page - 1, len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Split text into lines for processing
        lines = text.split('\n')

        for line in lines:

            if stop_word in line:
                found_stop_word = True
                if current_answer:
                    answers_list.append(' '.join(current_answer))
                current_answer = []
                continue

            # Check if the line is a question
            elif re.match(question_regex, line) or question_variety in line or re.match(r"^ \d*\. The answer is [A-Z]\. ", line):

                if question_variety in line:
                    continue

                # check if it's an answer:
                elif answer in line or re.match(r"^ \d*\. The answer is [A-Z]\. ", line): #
                    found_answer = True
                    if current_answer:
                        answers_list.append(' '.join(current_answer))
                    current_answer = [re.sub(r'^[\s]*\d+\.\s*', '', line)]
                    found_question = False
                    found_stop_word = False
                    continue

                found_question = True
                found_stop_word = False
                # if we continue to next question, add current one to the list
                if current_question:
                    questions_list.append(' '.join(current_question))
                current_question = [line]

            elif found_question and not found_stop_word:
                current_question.append(line)

            elif found_answer and not found_stop_word:
                current_answer.append(line)

    # problematic questions. will add later, now for the presentation don't need
    question_to_skip = [39, 40, 53, 54, 119, 120, 139, 140, 150, 151, 152, 153]
    # the pdf reader mixed up the questions so we trim it a bit
    questions_list = questions_list[:82] + [questions_list[92]] +  questions_list[88:92] + questions_list[93:]

    for question_index in question_to_skip:
        for question in questions_list:
            if question.startswith(str(question_index)):
                questions_list.remove(question)
                break
            else:
                index = questions_list.index(question)
                questions_list[index] = re.sub(r'^[\s]*\d+\.\s*', '', question)
# Convert the list of Q&A pairs into a DataFrame
    hematology_qa = pd.DataFrame({'Questions': questions_list, 'Answers': answers_list})

# Save the DataFrame to a CSV file
    hematology_qa.to_csv(csv_path, index=False)

    print(f'Successfully extracted Q&A to {csv_path}')


extract_qa_from_questionnaire(second_questionnaire_pdf_path, second_questionnaire_start_page, second_questionnaire_question_regex,
                              second_questionnaire_question_variety,
                              second_questionnaire_answer_id, second_questionnaire_stop_word, second_questionnaire_csv_path)