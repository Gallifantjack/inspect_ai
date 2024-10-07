from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.model import GenerateConfig


LANGUAGES = [
    "AR_XY",
    "BN_BD",
    "DE_DE",
    "ES_LA",
    "FR_FR",
    "HI_IN",
    "ID_ID",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "PT_BR",
    "SW_KE",
    "YO_NG",
    "ZH_CN",
]


# map records to inspect sample
def record_to_sample(record):
    return Sample(
        input=record["Question"],
        choices=[
            str(record["A"]),
            str(record["B"]),
            str(record["C"]),
            str(record["D"]),
        ],
        target=record["Answer"],
        metadata={"subject": record["Subject"]},
    )


MULTIPLE_CHOICE_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()


@task
def mmmlu_language(language: str, cot=False, temperature=0.1):
    if language not in LANGUAGES:
        raise ValueError(
            f"Invalid language. Please choose from: {', '.join(LANGUAGES)}"
        )

    dataset = hf_dataset(
        path="openai/MMMLU",
        name=language,
        split="test",
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True,
    )

    if cot:
        plan = multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE_COT)
    else:
        plan = multiple_choice()

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=choice(),
        config=GenerateConfig(temperature=temperature),
    )
