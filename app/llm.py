import json
import os

from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger


load_dotenv()


def parse_json_result(result):
    try:
        start_idx = result.index("{")
        end_idx = result.index("}") + 1
        return json.loads(result[start_idx:end_idx])
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing JSON result: {e}")
        return {}


def generate_personality_report(personality_traits):
    base_url = os.getenv("OLLAMA_URL")
    model = os.getenv("LLAMA_MODEL")
    llm = Ollama(base_url=base_url, model=model)

    system_prompt = """
        Analyse the following personality traits and generate a personality report based on it with 5 sentences in each field.
        Personality traits: {personality_traits}

        The output must follow the following JSON format:
            {{
                "overview": "overview of the personality as proper string enclosed in double quotes",
                "Intution": "intution of the personality as proper string enclosed in double quotes",
                "Patience": "patience of the personality as proper string enclosed in double quotes",
                "Flexibility": "flexibility of the personality as proper string enclosed in double quotes",
                "Outcome Orientation": "outcome orientation of the personality as proper string enclosed in double quotes",
                "Analytical Thinking": "analytical thinking of the personality as proper string enclosed in double quotes",
                "Active Listening": "active listening of the personality as proper string enclosed in double quotes",
                "Working Under Pressure": "working under pressure of the personality as proper string enclosed in double quotes",
                "Results Orientation:": "results orientation of the personality as proper string enclosed in double quotes",
                "Interpersonal Skills": "interpersonal skills of the personality as proper string enclosed in double quotes",
                "Attention to Detail": "attention to detail of the personality as proper string enclosed in double quotes",
                "Compliance": "compliance of the personality as proper string enclosed in double quotes",
                "Communication": "communication of the personality as proper string enclosed in double quotes",
                "Teamwork": "teamwork of the personality as proper string enclosed in double quotes",
                "Discipline": "discipline of the personality as proper string enclosed in double quotes",
                "Budgeting": "budgeting of the personality as proper string enclosed in double quotes",
                "Work Ethic": "work ethic of the personality as proper string enclosed in double quotes",
                "Delegation": "delegation of the personality as proper string enclosed in double quotes",
                "Dependability": "dependability of the personality as proper string enclosed in double quotes",
                "Decision Making": "decision making of the personality as proper string enclosed in double quotes",
                "Stress Management": "stress management of the personality as proper string enclosed in double quotes",
                Coordinatig: "coordinating of the personality as proper string enclosed in double quotes",
                "Change Management": "change management of the personality as proper string enclosed in double quotes",
                "Organizing": "organizing of the personality as proper string enclosed in double quotes",
                "Conflict Resolution": "conflict resolution of the personality as proper string enclosed in double quotes",
                "Detail Orientation": "detail orientation of the personality as proper string enclosed in double quotes",
                "Social Media": "social media of the personality as proper string enclosed in double quotes",
                "Consistency": "consistency of the personality as proper string enclosed in double quotes",
                "Questioning": "questioning of the personality as proper string enclosed in double quotes",
                "Verbal Communication": "verbal communication of the personality as proper string enclosed in double quotes",
                "Training": "training of the personality as proper string enclosed in double quotes",
                "Creativity": "creativity of the personality as proper string enclosed in double quotes",
                "Sense of Humor": "sense of humor of the personality as proper string enclosed in double quotes",
                "Self Motivation": "self motivation of the personality as proper string enclosed in double quotes",
                "Prioritization": "prioritization of the personality as proper string enclosed in double quotes",
                "Vision": "vision of the personality as proper string enclosed in double quotes",
                "Independence": "independence of the personality as proper string enclosed in double quotes",
                "Writing": "writing of the personality as proper string enclosed in double quotes",
                "Goal Setting": "goal setting of the personality as proper string enclosed in double quotes",
                "Inegrity": "integrity of the personality as proper string enclosed in double quotes",
                "Learning Agility": "learning agility of the personality as proper string enclosed in double quotes",
                "Networking": "networking of the personality as proper string enclosed in double quotes",
                "Transparency": "transparency of the personality as proper string enclosed in double quotes",
            }}
        Ensure that each field is described with atlest 5 sentences.
        Do not include any additional information or explanations outside of these fields.
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    result = chain.invoke({"personality_traits": json.dumps(personality_traits)})
    return parse_json_result(result)
