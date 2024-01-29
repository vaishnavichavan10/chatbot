import streamlit as st
#from langchain.llms import OpenAI
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os

# Set OpenAI API key
os.environ["OPEN_API_KEY"] = "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"

# Function to generate itinerary and hotel recommendations based on user inputs
def generate_itinerary(destination):
    # Initialize OpenAI language model
    llm = OpenAI(openai_api_key=os.environ["OPEN_API_KEY"], model="gpt-3.5-turbo-instruct", temperature=0.6)

    # Define destination prompt template
    destination_template = PromptTemplate(input_variables=['destination'], template="Please tell me your preferred {destination}")
    destination_chain = LLMChain(llm=llm, prompt=destination_template)

    # Define places prompt template
    places_template = PromptTemplate(input_variables=['destination'], template="Tell me some amazing places within {destination}")
    places_chain = LLMChain(llm=llm, prompt=places_template, output_key="itinerary")

    # Define shopping prompt template
    shopping_template = PromptTemplate(input_variables=['destination'], template="Suggest me some good places to shop and what to shop in {destination}")
    shopping_chain = LLMChain(llm=llm, prompt=shopping_template, output_key="shops")

    # Define hotel recommendation prompt template
    hotel_template = PromptTemplate(input_variables=['destination'], template="Recommend some hotels to stay in {destination}")
    hotel_chain = LLMChain(llm=llm, prompt=hotel_template, output_key="hotels")

    # Create a sequential chain combining all the chains
    chain = SequentialChain(chains=[destination_chain, places_chain, shopping_chain, hotel_chain],
                            input_variables=['destination'],
                            output_variables=["itinerary", 'shops', 'hotels'])

    # Create input dictionary with the required key 'input'
    inputs = {'destination': destination}

    # Invoke the chain to generate itinerary and hotel recommendations
    result = chain.invoke(inputs)

    return result

# Streamlit app
def main():
    # Change background color
    st.markdown(
        """
        <style>
            body {
                background-color: #0000ff; /* Set your preferred background color here */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add background image
    st.image("bg.jpg", use_column_width=True)

    # Set page title
    st.title("Travel Assistant")

    # Get user inputs
    destination = st.text_input("What is your destination?")

    # Button to generate itinerary and hotel recommendations
    if st.button("Generate Itinerary and Hotel Recommendations"):
        # Generate itinerary and hotel recommendations based on user inputs
        result = generate_itinerary(destination)

        # Display itinerary
        st.write("Day-wise Itinerary:")
        st.write(result['itinerary'])

        # Display shopping recommendations
        st.write("Shopping Recommendations:")
        st.write(result['shops'])

        # Display hotel recommendations
        st.write("Hotel Recommendations:")
        st.write(result['hotels'])

if __name__ == "__main__":
    main()
