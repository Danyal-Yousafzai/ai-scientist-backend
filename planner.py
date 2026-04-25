import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from qc_agent import run_literature_qc

load_dotenv()



# 1. Define the exact output structure for the Experiment Plan
class ExperimentPlan(BaseModel):
    protocol_steps: list[str] = Field(description="Step-by-step methodology grounded in real published protocols.")
    materials: list[dict] = Field(description="List of materials. Each dict should have 'item_name', 'catalog_number' (can be realistic estimates), and 'supplier'.")
    budget_estimate: str = Field(description="Realistic cost estimate with line items and currency.")
    timeline: dict[str, str] = Field(description="Phased breakdown with dependencies. Keys should be the timeframe (e.g., 'Week 1', 'Day 1'), values should be the task description.")

def generate_experiment_plan(hypothesis: str):
    # Step 1: Run the QC Check to get context
    qc_result, context = run_literature_qc(hypothesis)
    
    print("\nDrafting Full Experiment Plan (This may take 10-15 seconds)...")

    # Step 2: Set up the LLM for the heavy lifting
    # We are using Llama 3.1 8b, but if you have access, switching to "llama-3.3-70b-versatile" here makes the science much better
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    # Step 3: Set up the JSON parser
    parser = PydanticOutputParser(pydantic_object=ExperimentPlan)

    # Step 4: Prompt the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite Principal Investigator running a state-of-the-art lab. "
                   "Your job is to take a scientific hypothesis and generate an operationally realistic experiment plan that a lab could execute immediately. "
                   "Use the provided literature context to ground your protocol in reality. Be highly specific with reagents, cell lines, and standards. "
                   "\n\nYou must respond strictly in JSON format.\n{format_instructions}"),
        ("user", "Hypothesis: {hypothesis}\n\nLiterature Context:\n{context}\n\nNovelty Status: {novelty_signal}\n\n"
                 "Generate the full experiment plan.")
    ])

    # Step 5: Chain and Execute
    chain = prompt | llm | parser
    
    plan = chain.invoke({
        "hypothesis": hypothesis, 
        "context": context,
        "novelty_signal": qc_result.novelty_signal,
        "format_instructions": parser.get_format_instructions()
    })
    
    return {
        "qc_check": qc_result.model_dump(),
        "experiment_plan": plan.model_dump()
    }

# Test it locally
if __name__ == "__main__":
    test_hypothesis = "Replacing sucrose with trehalose as a cryoprotectant in the freezing medium will increase post-thaw viability of HeLa cells."
    
    final_output = generate_experiment_plan(test_hypothesis)
    
    print("\n=======================================================")
    print("                 FINAL EXPERIMENT PLAN                 ")
    print("=======================================================\n")
    print(json.dumps(final_output, indent=2))