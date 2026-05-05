import requests


SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer clearly and simply.
If document context is provided, use it carefully.
If the answer is not present in the document context, say that clearly.
Do not make up facts.
"""

DEFAULT_MODEL = "katanemo/Arch-Router-1.5B"


def _format_chat_history(chat_history):
    if not chat_history:
        return "No previous chat history."

    lines = []
    for message in chat_history[-8:]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def _format_context(vectorstore, user_input):
    if vectorstore is None:
        return "No document context available."

    docs = vectorstore.similarity_search(user_input, k=4)
    if not docs:
        return "No relevant document context found."

    return "\n\n".join(doc.page_content for doc in docs)


def _call_huggingface(hf_token, model_name, temperature, max_new_tokens, prompt):
    selected_model = model_name.strip() if model_name and model_name.strip() else DEFAULT_MODEL

    api_url = f"https://router.huggingface.co/hf-inference/models/{selected_model}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(f"{response.status_code} {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def build_chat_chain(hf_token, model_name, temperature, max_new_tokens, vectorstore=None):
    def invoke(payload):
        user_input = payload["input"]
        chat_history = payload.get("chat_history", [])

        history_text = _format_chat_history(chat_history)
        context_text = _format_context(vectorstore, user_input)

        prompt = f"""
Chat History:
{history_text}

Document Context:
{context_text}

User Question:
{user_input}

Instructions:
- Answer in a clean and easy way.
- If the user asks for points, give points.
- If context is missing, say so clearly.
"""

        answer = _call_huggingface(
            hf_token=hf_token,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
        )

        return {"answer": answer}

    return type("SimpleHFChain", (), {"invoke": staticmethod(invoke)})()
