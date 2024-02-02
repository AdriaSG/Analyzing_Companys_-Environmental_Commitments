import pandas as pd
import streamlit as st
import utils

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS

df = pd.read_csv("extracted_text_verified_urls_2023_labeled_07.10.23.csv")
sample = df[["name", "url"]].sample(25)

st.set_page_config(
    page_title="Climate Pledges",
    page_icon="🌎",
)

# Initialize the LLM & Embeddings n_ctx=3584,
with st.spinner("Loading LLM"):
    embeddings, llm = utils.define_embeddings_llm()

st.title("🌎 Analyzing Company's Environmental Commitments")
st.subheader("1. Get text:")
url = st.text_input(
    "Provide a URL from a Climate Pledge:",
    value="https://www.adyen.com/social-responsibility/climate#neutral",
)
st.caption("*Supported files: HTML and PDF")
with st.expander("Show more example urls:"):
    st.dataframe(sample)




response_schemas = [
    ResponseSchema(name="answer", description="Yes or No as answer of user question."),
    ResponseSchema(name="support", description="Support your answer in 2 sentences."),
    ResponseSchema(
        name="source",
        description="Quote text from given context.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if url:
    # Split
    with st.spinner("Loading text from document..."):
        extracted_text = utils.load_text_from_url(url)
    st.success(f"Your document had been loaded.")

    st.subheader("2. Create Embeddings:")
    with st.spinner("Creating embeddings..."):
        vectordb = FAISS.from_documents(
            documents=utils.split_text(
                extracted_text, chunk_size=440, chunk_overlap=25
            ),
            embedding=embeddings,
        )
    st.success(f"Embeddings created successfully.")
    st.subheader("3. Narrow Context:")
    question = st.text_input(
        "Enter a question:",
        value="Has the company made a commitment to reduce Scope 1, 2 or 3 emissions?",
    )
    if question:
        with st.spinner("Getting relevant document chunks to the query."):
            # Retriever A:
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=vectordb.as_retriever(), llm=llm
            )
            results = retriever_from_llm.get_relevant_documents(question)
            st.write(f"Found {len(results )} relevant chunks")
            with st.expander("Show relevant chunks:"):
                st.write(format_docs(results))
        st.subheader("4. Create an answer:")
        with st.spinner("Creating an answer..."):
            generic_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer
                                    the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer 
                                    concise.<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]""",
            )
            output_parser_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer
                                    the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer 
                                    concise.{format_instructions}<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]""",
                partial_variables={
                    "format_instructions": format_instructions,
                },
                output_parser=output_parser,
            )

            chain = generic_prompt | llm
            parsed_chain = output_parser_prompt | llm
            answer = chain.invoke(
                {"context": format_docs(results), "question": question}
            )
            st.write("Answer:")
            st.write(answer)

            parsed_answer = parsed_chain.invoke(
                {"context": format_docs(results), "question": question}
            )
            st.write("Parsed_answer:")
            st.write(parsed_answer)
    else:
        st.empty()
else:
    st.empty()
