import evaluate
import pandas as pd
import numpy as np

# read the csv with the answers
first_questionnaire_experiment = pd.read_csv("./Experiment.csv")

# let's get the each of the generated summaries
actual_answers = first_questionnaire_experiment['Answers']
pre_rag_answers = first_questionnaire_experiment['chatGpt3.5']
post_rag_answers = first_questionnaire_experiment['AnswersWithRag']

# remember, ROUGE isn't perfect but it does indicate the overall increase in summarization that we have accomplished by using RAG

rouge = evaluate.load('rouge')

# zipped_summaries = list(zip(actual_answers, pre_rag_answers, post_rag_answers))
#
# ep = pd.DataFrame(zipped_summaries, columns=['actual_answers', 'pre_rag_answers', 'post_rag_answers'])

pre_rag_model_results = rouge.compute(
    predictions=pre_rag_answers,
    references=actual_answers,
    use_aggregator=True,
    use_stemmer=True,
)

post_rag_model_results = rouge.compute(
    predictions=post_rag_answers,
    references=actual_answers,
    use_aggregator=True,
    use_stemmer=True,
)

print('pre_rag_answers:')
print(pre_rag_model_results)
print('post_rag_answers:')
print(post_rag_model_results)

print("\n###################### Absolute percentage improvement of post_rag_answers over pre_rag_answers #################################\n")


overall_improvement = (np.array(list(post_rag_model_results.values())) - np.array(list(pre_rag_model_results.values())))
for key, value in zip(post_rag_model_results.keys(), overall_improvement):
    print(f'{key}: {value*100:.2f}%')
