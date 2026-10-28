import time

from dotenv import load_dotenv
from openai import OpenAI
from typesafe_sdk import Noul, TypeSafeClient

load_dotenv()

MESSAGE = "I was charged twice for my subscription this month."

start = time.perf_counter()
response = TypeSafeClient().system_one(
    state=MESSAGE,
    questions={"billing": Noul(instructions="The message is about billing")},
)
jev_time = time.perf_counter() - start

start = time.perf_counter()
completion = OpenAI().chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": f'"{MESSAGE}" Is this about billing?'}],
)
llm_time = time.perf_counter() - start

print(f"jev: {jev_time * 1000:.0f} ms -> {response.nouls['billing'].noul:.2f}")
print(f"llm: {llm_time * 1000:.0f} ms -> {completion.choices[0].message.content.strip()}")
