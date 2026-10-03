import json

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric

from SRC.retrieval import build_retriever
from evals.groq_judge import GroqQwenJudge

load_dotenv()

golden_path = "goldenDATA/golden_dataset.json"
threshold = 0.7
judge_model = GroqQwenJudge()

# loading golden dataset 
with open(golden_path) as f:
    golden = json.load(f)

retriever = build_retriever()
test_cases = []


for g in golden:
    retrieved = retriever.invoke(g['query'])
    retrieved_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g['query'],
            expected_output=g['ideal_answer'],
            retrieval_context=retrieved_context
        ))

metrics = [
    ContextualRecallMetric(threshold=threshold,include_reason=True,model=judge_model),
    ContextualPrecisionMetric(threshold=threshold,include_reason=True,model=judge_model)
]

# evaluation 
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
            "retriever": "reranker",          # vs "reranked" when you swap it in , basek5
            "embedding_model": "HuggingFaceEmbeddings",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 3,
            "judge_model": judge_model,
            "golden_set": golden,
        },
        
)