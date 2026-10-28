"""Answer a question from retrieved context."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

GENERATOR_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

prompt = ChatPromptTemplate.from_template(
    "Answer the question below. Always end the answer with - Hope this answers your question \n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=TEMPERATURE)

chain = prompt | llm | StrOutputParser()


def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


if __name__ == "__main__":
    query = "What if my card is lost?"

    # Manual context: tests the generator on its own, no Chroma, no embeddings.
    context = [
        "Section 2 - Cards. If your card is lost or stolen, freeze it immediately in the "
        "app under Cards > Freeze card, then call us on 0800 123 4567. We will cancel the "
        "old card and post a replacement within 5 working days.",
        "Section 2 - Cards. You are not liable for any transactions made after you report "
        "the card lost. Transactions before the report may be covered if you did not act "
        "negligently.",
    ]

    # End to end through the real retriever instead:
    # from retriever import get_retriever
    # context = [doc.page_content for doc in get_retriever().invoke(query)]

    print(generate(query, context))
