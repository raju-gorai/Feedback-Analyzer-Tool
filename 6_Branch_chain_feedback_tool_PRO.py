from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import streamlit as st


def build_chain():
    model = ChatNVIDIA(model="openai/gpt-oss-20b")
    parser = StrOutputParser()

    class Feedback(BaseModel):
        sentiment: Literal["positive", "negative"] = Field(
            description="Classify the sentiment of the review as either positive or negative"
        )

    parser2 = PydanticOutputParser(pydantic_object=Feedback)

    prompt1 = PromptTemplate(
        template=(
            "classify the sentiment of the following review as either positive or negative \n {feedback} \n "
            "{format_instructions}"
        ),
        input_variables=["feedback"],
        partial_variables={"format_instructions": parser2.get_format_instructions()},
    )

    classifier_chain = prompt1 | model | parser2

    prompt2 = PromptTemplate(
        template="Write an appropriate response to the following review in 30 words:\n {feedback}",
        input_variables=["feedback"],
    )

    prompt3 = PromptTemplate(
        template=(
            "Write an appropriate response to the following review in 30 words and ask to contact us at support@aitool.com\n {feedback}"
        ),
        input_variables=["feedback"],
    )

    branch_chain = RunnableBranch(
        (lambda x: x.sentiment == "positive", prompt2 | model | parser),
        (lambda x: x.sentiment == "negative", prompt3 | model | parser),
        RunnableLambda(lambda x: "Invalid sentiment"),
    )

    return classifier_chain | branch_chain


def run_feedback(review_text: str):
    chain = build_chain()
    return chain.invoke({"feedback": review_text})


def streamlit_app():
    st.set_page_config(page_title="Feedback Response Generator", layout="centered")
    st.title("Feedback Analyzer Tool")
    st.write(
        "Enter a customer review below to classify sentiment and generate a tailored response."
    )

    review_text = st.text_area(
        "Review text",
        value="The movie was very good.",
        height=150,
    )

    if st.button("Generate response"):
        if not review_text.strip():
            st.warning("Please enter a review before generating a response.")
        else:
            with st.spinner("Analyzing review..."):
                try:
                    result = run_feedback(review_text)
                    st.success("Response generated")
                    st.subheader("Model output")
                    st.write(result)
                except Exception as exc:
                    st.error(f"Error running the chain: {exc}")


if __name__ == "__main__":
    if st is not None:
        streamlit_app()
    else:
        print(
            "Streamlit is not installed."
        )
    