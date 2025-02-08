import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)
