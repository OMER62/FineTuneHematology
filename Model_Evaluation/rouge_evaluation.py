import evaluate
import pandas as pd
import numpy as np

# read the csv with the answers
first_questionnaire_experiment = pd.read_csv("./Experiment.csv")
second_questionnaire_experiment = pd.read_csv("./Experiment2.csv")
# let's get the each of the generated summaries
actual_answers_first_experiment = first_questionnaire_experiment['Answers']
pre_rag_answers_first_experiment = first_questionnaire_experiment['chatGpt3.5']
post_rag_answers_first_experiment = first_questionnaire_experiment['AnswersWithRag']

actual_answers_second_experiment = second_questionnaire_experiment['Answers']
pre_rag_answers_second_experiment = second_questionnaire_experiment['chatGpt3.5']
post_rag_answers_second_experiment = second_questionnaire_experiment['AnswersWithRag']

# remember, ROUGE isn't perfect but it does indicate the overall increase in summarization that we have accomplished by using RAG

rouge = evaluate.load('rouge')

# zipped_summaries = list(zip(actual_answers, pre_rag_answers, post_rag_answers))
#
# ep = pd.DataFrame(zipped_summaries, columns=['actual_answers', 'pre_rag_answers', 'post_rag_answers'])

pre_rag_model_results_first_experiment = rouge.compute(
    predictions=pre_rag_answers_first_experiment,
    references=actual_answers_first_experiment,
    use_aggregator=True,
    use_stemmer=True,
)

post_rag_model_results_first_experiment = rouge.compute(
    predictions=post_rag_answers_first_experiment,
    references=actual_answers_first_experiment,
    use_aggregator=True,
    use_stemmer=True,
)

pre_rag_model_results_second_experiment = rouge.compute(
    predictions=pre_rag_answers_second_experiment,
    references=actual_answers_second_experiment,
    use_aggregator=True,
    use_stemmer=True,
)

post_rag_model_results_second_experiment = rouge.compute(
    predictions=post_rag_answers_second_experiment,
    references=actual_answers_second_experiment,
    use_aggregator=True,
    use_stemmer=True,
)
print(f"First Experiment number of questions: {len(actual_answers_first_experiment)}\n")
print('pre_rag_answers_first_experiment:')
print(pre_rag_model_results_first_experiment)
print('post_rag_answers_first_experiment:')
print(post_rag_model_results_first_experiment)

print("\n Absolute percentage improvement of post_rag_answers over pre_rag_answers First Experiment dataset\n")


overall_improvement_first_experiment = (np.array(list(post_rag_model_results_first_experiment.values())) \
                                        - np.array(list(pre_rag_model_results_first_experiment.values())))
for key, value in zip(post_rag_model_results_first_experiment.keys(), overall_improvement_first_experiment):
    print(f'{key}: {value*100:.2f}%')
print(f"\nSecond Experiment number of questions: {len(actual_answers_second_experiment)}\n")
print('pre_rag_answers_second_experiment:')
print(pre_rag_model_results_second_experiment)
print('post_rag_answers_second_experiment:')
print(post_rag_model_results_second_experiment)

print("\n Absolute percentage improvement of post_rag_answers over pre_rag_answers Second Experiment dataset\n")


overall_improvement_second_experiment = (np.array(list(post_rag_model_results_second_experiment.values())) \
                                         - np.array(list(pre_rag_model_results_second_experiment.values())))
for key, value in zip(post_rag_model_results_second_experiment.keys(), overall_improvement_second_experiment):
    print(f'{key}: {value*100:.2f}%')
