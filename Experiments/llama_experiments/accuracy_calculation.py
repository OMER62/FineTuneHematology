import pandas as pd
from sklearn.metrics import accuracy_score
def accuracy(df):
    answers = df['Answers']
    llm_answers = df['llama2_Base']
    llm_answers = [answer[2] for answer in llm_answers]

    rag_answers = df['llama_Rag']
    rag_answers = [answer[2] for answer in rag_answers]
    
    pipeline_answers = df['llama_Qpipeline']
    pipeline_answers = [answer[2] for answer in pipeline_answers]
    results = {
        'llama_base' : [f"{accuracy_score(answers, llm_answers):.2f}"],
        'llama_RAG' : [f"{accuracy_score(answers, rag_answers):.2f}"],
        'llama_pipeline' : [f"{accuracy_score(answers, pipeline_answers):.2f}"],
    }
    print(results)
    pd.DataFrame(results).to_csv("Accuracy_calculations_high.csv")

if __name__ == "__main__":
    df = pd.read_csv("llama2-7b_hematology-index-llama-high_piplineAndRag.csv")
    accuracy(df)
    # df["Answers"] = [answer.split("The answer is ")[1][0] for answer in answers]
    # df.to_csv("llama2-7b_hematology-index-llama-high_piplineAndRag.csv")
    