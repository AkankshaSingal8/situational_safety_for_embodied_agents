import os, base64
import vlm_prompt_runner.config
from vlm_prompt_runner.episode import load_episode
from vlm_prompt_runner.prompt_loader import load_prompt
from vlm_prompt_runner.runner import build_prompt
import openai

ep = load_episode("vlm_inputs/safelibero_spatial/level_I/task_0/episode_00")
prompt = build_prompt(
    load_prompt("prompts/obstacle_id/p8_ranked_collision.md"),
    ep["task_description"],
    {
        "task_description": ep["task_description"],
        "object_list": ep.get("object_list", ""),
        "object_list_with_positions": ep.get("object_list_with_positions", ""),
    },
)

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

content = []
for p in [ep["agentview"], ep["eye_in_hand"], ep["backview"]]:
    b64 = base64.standard_b64encode(open(p, "rb").read()).decode()
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
content.append({"type": "text", "text": prompt})

resp = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": content}],
    max_completion_tokens=1024,
)
msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("content repr:", repr(msg.content))
print("refusal:", repr(msg.refusal))
print("content preview:", (msg.content or "")[:500])
