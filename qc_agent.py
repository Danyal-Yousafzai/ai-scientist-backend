import os
from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# 1. Define the exact output structure for the QC check
class QCResult(BaseModel):
    novelty_signal: str = Field(description="Must be exactly one of: 'not found', 'similar work exists', or 'exact match found'")
    references: list[str] = Field(description="List of 1-3 relevant references (Title and URL) if applicable.")

def run_literature_qc(hypothesis: str):
    print(f"Running Literature QC for: {hypothesis}...")
    
    # 2. Fetch academic context using Tavily
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    search_query = f"Scientific research, papers, and lab protocols for: {hypothesis}"
    tavily_response = tavily.search(query=search_query, search_depth="advanced", max_results=4)
    
    context = "\n".join([f"- {res['title']}: {res['content']} ({res['url']})" for res in tavily_response['results']])

    # 3. Set up the LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # 4. Use PydanticOutputParser (This fixes the Groq Llama tool-calling bug)
    parser = PydanticOutputParser(pydantic_object=QCResult)

    # 5. Prompt the LLM, injecting the format instructions directly into the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior scientific researcher. Your job is to act as a plagiarism and novelty checker for scientific hypotheses. "
                   "Read the provided search context and determine if the exact protocol or very similar work has been done before.\n\n"
                   "You must respond strictly in JSON format.\n{format_instructions}"),
        ("user", "Hypothesis: {hypothesis}\n\nSearch Context:\n{context}")
    ])

    # 6. Chain the prompt -> LLM -> Parser together
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "hypothesis": hypothesis, 
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })
    
    return result, context

# Test it locally
if __name__ == "__main__":
    test_hypothesis = "Replacing sucrose with trehalose as a cryoprotectant in the freezing medium will increase post-thaw viability of HeLa cells."
    qc_result, raw_context = run_literature_qc(test_hypothesis)
    print("\n--- QC Result ---")
    print(qc_result.model_dump_json(indent=2))