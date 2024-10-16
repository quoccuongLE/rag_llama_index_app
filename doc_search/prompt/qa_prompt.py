def get_context_prompt(language: str) -> str:
    if language == "vie":
        return CONTEXT_PROMPT_VI
    return CONTEXT_PROMPT_EN


def get_system_prompt(language: str, is_rag_prompt: bool = True) -> str:
    if language == "vie":
        return SYSTEM_PROMPT_RAG_VI if is_rag_prompt else SYSTEM_PROMPT_VI
    return SYSTEM_PROMPT_RAG_EN if is_rag_prompt else SYSTEM_PROMPT_EN


SYSTEM_PROMPT_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

SYSTEM_PROMPT_RAG_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

CONTEXT_PROMPT_EN = """\
Here are the relevant documents for the context:

{context_str}

Instruction: Based on the above documents, provide a detailed answer for the user question below. \
Answer 'don't know' if not present in the document."""

CONDENSED_CONTEXT_PROMPT_EN = """\
Given the following conversation between a user and an AI assistant and a follow up question from user,
rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:\
"""

SYSTEM_PROMPT_VI = """\
Đây là một cuộc trò chuyện giữa người dùng và một trợ lí trí tuệ nhân tạo. \
Trợ lí đưa ra các câu trả lời hữu ích, chi tiết và lịch sự đối với các câu hỏi của người dùng dựa trên bối cảnh. \
Trợ lí cũng nên chỉ ra khi câu trả lời không thể được tìm thấy trong ngữ cảnh. \
Tất cả các câu trả lời phải bằng tiếng Việt"""

SYSTEM_PROMPT_RAG_VI = """\
Đây là một cuộc trò chuyện giữa người dùng và một trợ lí trí tuệ nhân tạo. \
Trợ lí đưa ra các câu trả lời hữu ích, chi tiết và lịch sự đối với các câu hỏi của người dùng dựa trên bối cảnh. \
Trợ lí cũng nên chỉ ra khi câu trả lời không thể được tìm thấy trong ngữ cảnh. \
Tất cả các câu trả lời phải bằng tiếng Việt"""

CONTEXT_PROMPT_VI = """\
Dưới đây là các tài liệu liên quan cho ngữ cảnh:

{context_str}

Hướng dẫn: Dựa trên các tài liệu trên, cung cấp một câu trả lời chi tiết cho câu hỏi của người dùng dưới đây. \
Trả lời 'không biết' nếu không có trong tài liệu."""

CONDENSED_CONTEXT_PROMPT_VI = """\
Cho cuộc trò chuyện sau giữa một người dùng và một trợ lí trí tuệ nhân tạo và một câu hỏi tiếp theo từ người dùng,
đổi lại câu hỏi tiếp theo để là một câu hỏi độc lập.

Lịch sử Trò chuyện:
{chat_history}
Đầu vào Tiếp Theo: {question}
Câu hỏi độc lập:\
"""

_EXAMPLE_INSTRUCTION_PROMPT = """
This document is an insurance policy.
When a benefits/coverage/exlusion is describe in the document ammend to it add a text in the follwing benefits string format (where coverage could be an exclusion).

For {nameofrisk} and in this condition {whenDoesThecoverageApply} the coverage is {coverageDescription}. 
                                        
If the document contain a benefits TABLE that describe coverage amounts, do not ouput it as a table, but instead as a list of benefits string.
                                       
"""

qa_template = """
    Please provide an answer based solely on the provided sources. 
    When referencing information from a source, "
    cite the appropriate source(s) using their corresponding numbers.
    Every answer should include at least one source citation. 
    Only cite a source when you are explicitly referencing it via page number.
    If none of the sources are helpful, you should indicate that.
    For example:\n
    Source 1 - Page 25:\n
    The sky is red in the evening and blue in the morning.\n
    Source 2 - Page 18:\n
    Water is wet when the sky is red.\n
    Query: When is water wet?\n
    Answer: Water will be wet when the sky is red [2], 
    which occurs in the evening [1].\n
    Now it's your turn. Below are several numbered sources of information:
    \n------\n
    {context_str}
    \n------\n
    Answer:
"""

summarization_template = (
    "Your are an expert in the field of {expert_domain_str}."
    "The following passage is an extract of from a scientific paper. You must"
    " summarize it without using any prior knowledge."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

job_skill_retrieval_template = (
    "Your are an excellent recruiter in the domain of {expert_domain_str}."
    "Your company is hiring for a position of {job_name}."
    "The text below is the job description."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
    "Please give the most essential requirements for the ideal candidate."
)


cover_letter_template_given_candidate_bio = (
    "Given some relevant information of a candidate listed below:"
    "\n--------------------\n"
    "{qualifications_str}"
    "\n--------------------\n"
    "Write a cover letter for a position of {job_name} to impress the hiring manager. "
    "Use only skills he has. Do not mention skills or experiences that he does not have. The job description is detailed below:"
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

# multiple select
multi_select_item_in_resume = (
    "A resumé of a candidate if given below in Markdown format.\n"
    "---------------------\n"
    "{resume}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "of the most relevant qualifications and experiences from his résumé."
    "(no more than {max_outputs}, mention also where he works to earn those experiences in output) "
    "for the job description below:\n"
    "---------------------\n"
    "{job_description}"
    "---------------------\n"
)
