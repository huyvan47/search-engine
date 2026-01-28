def summarize_to_fact(client, conversation_text):
    prompt = f"""
Tóm tắt hội thoại sau thành các FACT ổn định lâu dài.
Chỉ trả JSON list.

HỘI THOẠI:
\"\"\"{conversation_text}\"\"\"

FORMAT:
[
  {{"type":"profile|preference|workflow","fact":"...","confidence":0.0}}
]
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content
