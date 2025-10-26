import json
import re
from copy import deepcopy

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers.utils import _marshal_llm_to_json
from llama_index.core.schema import Document
from llama_index.core.settings import Settings

from doc_search.translator import Language

from .multilingual_base import MultiLingualBaseReader

info_extraction_template = (
    "The following passage is a job description."
    "You must extract all information about {information_str} without adding prior knowledge"
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)


def parse_time_period(text_line: str) -> tuple[str, str]:
    """
    Parses a time period from a given text line.

    Args:
      text_line: The text line to parse.

    Returns:
      A tuple containing the start and end dates (strings) if a valid
      time period is found, otherwise None.
    """

    # Define patterns for different time period formats
    patterns = [
        r"(\d{1,2}/\d{1,2}/\d{4}) - (\d{1,2}/\d{1,2}/\d{4})",  # MM/DD/YYYY - MM/DD/YYYY
        r"(\d{1,2}/\d{1,2}/\d{4}) - (Present)",
        r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{4} - (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}",
        r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{4} - (Present)",
        r"(\d{4}) - (\d{4})",  # YYYY - YYYY
        r"(\d{4}) - (Present)",  # YYYY - Present
    ]

    for pattern in patterns:
        match = re.search(pattern, text_line)
        if match:
            try:
                start_date, end_date = tuple(match.group(0).split("-"))
            except:
                start_date = match.group(0)
                end_date = None

            return start_date.strip(), end_date.strip()

    return None


class JobDescriptionReader(MultiLingualBaseReader):

    def __init__(
        self,
        tgt_language: Language = Language("eng"),
        translator_config: dict | None = None,
    ):
        super().__init__(tgt_language, translator_config)
        self._llm = Settings.llm
        self._template = info_extraction_template
        self._key_items = {
            "skill and qualification requirements": {
                "Education": "Required education for the position",
                "Professional Experience": "Required experience for the position",
                "Technical Skills": "Required technical skills for the position",
                "Soft skills": "Required soft skills for the position",
                "Nice to Haves": "Other requirements that make candidate outstanding",
            },
            "about the job": {
                "job name": "Title of the hiring position",
                "responsibilities": "All responsibilities for the position",
            },
            "about the company": {
                "company name": "The name of the company which is hiring"
            },
            "about the team and project": {
                "environment": "General information about the team and project in which the candicate will work"
            },
        }
        self._retrieved_infos = self._init_items()

    def _init_items(self) -> dict:
        info = dict()
        for topic, details in self._key_items.items():
            info[topic] = {k: None for k in details.keys()}
        return info

    def parse(self, text: str):
        for topic, details in self._key_items.items():
            text_with_template = self._template.format(
                information_str=topic, context_str=text
            )
            response_schemas = [
                ResponseSchema(name=k, description=v) for k, v in details.items()
            ]
            lc_output_parser = StructuredOutputParser.from_response_schemas(
                response_schemas
            )
            output_parser = LangchainOutputParser(lc_output_parser)
            final_query = output_parser.format(text_with_template)
            response = self._llm.complete(final_query)
            info_json = self._parse(response.text)
            self._retrieved_infos[topic].update(info_json)
        return

    def _parse(self, output: str) -> dict:
        """Attempts to parse the output of an LLM (Large Language Model) response into a JSON object.

        If the output can be directly parsed into JSON, it will return the JSON object. If the output contains a JSON-formatted string, it will extract and return the JSON object from that string. If neither of these approaches work, it will attempt to construct a JSON object from the output text.

        Returns:
            dict: A dictionary representing the parsed JSON object.
        """
        try:
            json_string = _marshal_llm_to_json(output)
            json_obj = json.loads(json_string)
            if not json_obj:
                raise ValueError(f"Failed to convert output to JSON: {output!r}")
            return json_obj
        except:
            return {}


class BaseMarkdownPortfolioReader(MultiLingualBaseReader):
    def __init__(
        self,
        tgt_language: Language = Language("eng"),
        translator_config: dict | None = None,
    ):
        super().__init__(tgt_language, translator_config)
        self._key_items = {
            "Summary": [],
            "Professional Experience": [],
            "Education": [],
            "Technical Skills": [],
            "Scholarship & Award": [],
            "Volunteering": [],
            "Publications & Patents": [],
            "Hobbies": [],
            "Research project": [],
            "Language": [],
        }
        self._item_structure = {
            "Summary": {},
            "Technical Skills": {},
            "Professional Experience": {
                3: ["position"],
                4: ["period", "place"],
                -1: ["Technology"],
            },
            "Education": {
                3: ["position"],
                4: ["period", "place"],
                -1: ["Keywords"],
            },
            "Scholarship & Award": {3: ["period", "award"]},
            "Volunteering": {
                3: ["position"],
                4: ["period", "place"],
            },
            "Publications & Patents": {},
            "Hobbies": {},
            "Research project": {4: ["name"]},
            "Language": {3: ["name"]},
        }
        self._retrieved_text_templates = {
            "Education": {
                "template": "Earned {position} from {place} in {period}",
                "fields": ["position", "place", "period"],
            },
            "Professional Experience": {
                "template": "Worked as {position} at {place} from {period_start} to {period_end}",
                "fields": ["position", "place", "period"],
            },
            "Technical Skills": {
                "template": "Proficient in {Technology}",
                "fields": ["Technology"],
            },
        }

    def generate_text(self, key: str) -> str:
        template = self._retrieved_text_templates[key]["template"]
        fields = self._retrieved_text_templates[key]["fields"]
        items = []
        for item in self._key_items[key]:
            if "Technical Skills" == key:
                skills = self.generate_tech_skills()
                items.extend([template.format(Technology=skill) for skill in skills])
            else:
                sub_dict = {k: item[k] for k in fields}
                if "Education" == key:
                    sub_dict["period"] = sub_dict["period"][1]
                    items.append(template.format(**sub_dict))
                elif "Professional Experience" == key:
                    if "period" in sub_dict.keys():
                        sub_dict["period_start"], sub_dict["period_end"] = sub_dict["period"]
                        sub_dict.pop("period")
                    items.append(template.format(**sub_dict))

        return items

    def generate_tech_skills(self) -> str:
        temp = self._key_items["Technical Skills"][0].split("\n")
        temp = [x for x in temp if x]
        skill_list = []
        for tmp in temp:
            raw_skills = tmp.split(":")[1].strip().split(",")
            skill_list.extend(raw_skills)
        return skill_list

    def parse(self, markdown_text: str, chunking: bool = True):
        lines = markdown_text.split("\n")
        topic = ""
        period: tuple[str] = ()
        current_item = {}
        current_text: str = ""
        for line in lines:
            header_match = re.match(r"^#+\s", line)
            if header_match:
                if current_text not in ["", "\n"] and topic != "":
                    if len(self._item_structure[topic]) == 0:
                        self._key_items[topic] = [current_text]
                        current_text = ""
                        current_item = {}
                    else:
                        current_item["content"] = [current_text]
                        self._key_items[topic].append(deepcopy(current_item))
                        current_text = ""
                        current_item = {}

                header_level = line.count("#")
                header_text = line.replace("#", "").strip()
                if header_level == 2 and header_text in self._key_items.keys():
                    topic = header_text

                elif header_level > 2:
                    if topic not in self._item_structure.keys():
                        continue
                    if header_level not in self._item_structure[topic].keys():
                        continue
                    keys = self._item_structure[topic][header_level]
                    for k in keys:
                        if k == "period":
                            period = parse_time_period(header_text)
                            current_item[k] = period

                            try:
                                period_str = f"({period[0]} - {period[1]})"
                                header_text = header_text.replace(period_str, "")
                            except:
                                try:
                                    header_text = header_text.split(":")[1]
                                except:
                                    pass
                        else:
                            current_item[k] = header_text.strip()
            else:
                if chunking:
                    current_text += line + "\n"
                else:
                    current_text.append(line)

    def get_item(self, key):
        return self._key_items[key]
