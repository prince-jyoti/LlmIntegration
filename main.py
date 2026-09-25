from dotenv import load_dotenv
import asyncio
from client import LLMClient

load_dotenv()

async def main():

    client = LLMClient.from_env()

    # ── Single completion ──────────────────────────────────────────
    response = await client.complete(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        prompt="What is a monad? Answer in 2 sentences.",
        system="You are a concise teacher.",
        max_tokens=200,
        temperature=0.3,
    )
    print(f"[{response.provider}] {response.content}")
    print(f"Tokens: {response.input_tokens}→{response.output_tokens}  "
          f"Latency: {response.latency_ms:.0f}ms")

    # ── Streaming ──────────────────────────────────────────────────
    print("\nStreaming from OpenAI:")
    async for token in client.stream("openai", "gpt-4o-mini", "Count to 5."):
        print(token, end="", flush=True)
    print()

    # ── Local Ollama ───────────────────────────────────────────────
    local = await client.complete("ollama", "llama3", "Hello!")
    print(f"\n[ollama] {local.content}")

    # ── Health check ───────────────────────────────────────────────
    status = await client.health_check_all()
    print("\nProvider health:", status)


asyncio.run(main())



# =========For testing purpose=======

#     # ── STEP 1: Health check all providers first ──────────────────
#     print("=" * 50)
#     print("STEP 1 — Checking all providers are reachable")
#     print("=" * 50)
#     status = await client.health_check_all()
#     for provider, healthy in status.items():
#         mark = "OK" if healthy else "FAIL"
#         print(f"  {provider:12} →  {mark}")
#     print()
#
#     # ── STEP 2: Test Ollama (free, local, test this first) ────────
#     print("=" * 50)
#     print("STEP 2 — Ollama (local)")
#     print("=" * 50)
#     try:
#         response = await client.complete(
#             provider    = "ollama",
#             model       = "llama3",
#             prompt      = "Say hello in one sentence.",
#         )
#         print(f"  Response   : {response.content}")
#         print(f"  Model      : {response.model}")
#         print(f"  Tokens in  : {response.input_tokens}")
#         print(f"  Tokens out : {response.output_tokens}")
#         print(f"  Latency    : {response.latency_ms:.0f}ms")
#         print("  STATUS     : PASSED")
#     except Exception as e:
#         print(f"  STATUS     : FAILED — {e}")
#     print()
#
#     # ── STEP 3: Test Anthropic complete ──────────────────────────
#     print("=" * 50)
#     print("STEP 3 — Anthropic complete()")
#     print("=" * 50)
#     try:
#         response = await client.complete(
#             provider    = "anthropic",
#             model       = "claude-haiku-4-5-20251001",
#             prompt      = "Say hello in one sentence.",
#             system      = "You are a helpful assistant.",
#             max_tokens  = 50,
#         )
#         print(f"  Response   : {response.content}")
#         print(f"  Model      : {response.model}")
#         print(f"  Tokens in  : {response.input_tokens}")
#         print(f"  Tokens out : {response.output_tokens}")
#         print(f"  Latency    : {response.latency_ms:.0f}ms")
#         print("  STATUS     : PASSED")
#     except Exception as e:
#         print(f"  STATUS     : FAILED — {e}")
#     print()
#
#     # ── STEP 4: Test OpenAI complete ─────────────────────────────
#     print("=" * 50)
#     print("STEP 4 — OpenAI complete()")
#     print("=" * 50)
#     try:
#         response = await client.complete(
#             provider    = "openai",
#             model       = "gpt-4o-mini",
#             prompt      = "Say hello in one sentence.",
#             system      = "You are a helpful assistant.",
#             max_tokens  = 50,
#         )
#         print(f"  Response   : {response.content}")
#         print(f"  Model      : {response.model}")
#         print(f"  Tokens in  : {response.input_tokens}")
#         print(f"  Tokens out : {response.output_tokens}")
#         print(f"  Latency    : {response.latency_ms:.0f}ms")
#         print("  STATUS     : PASSED")
#     except Exception as e:
#         print(f"  STATUS     : FAILED — {e}")
#     print()
#
#     # ── STEP 5: Test Anthropic streaming ─────────────────────────
#     print("=" * 50)
#     print("STEP 5 — Anthropic stream()")
#     print("=" * 50)
#     try:
#         print("  Tokens : ", end="", flush=True)
#         async for token in client.stream(
#             provider = "anthropic",
#             model    = "claude-haiku-4-5-20251001",
#             prompt   = "Count 1 to 5 slowly.",
#             max_tokens = 50,
#         ):
#             print(token, end="", flush=True)
#         print()
#         print("  STATUS : PASSED")
#     except Exception as e:
#         print(f"\n  STATUS : FAILED — {e}")
#     print()
#
#     # ── STEP 6: Test OpenAI streaming ────────────────────────────
#     print("=" * 50)
#     print("STEP 6 — OpenAI stream()")
#     print("=" * 50)
#     try:
#         print("  Tokens : ", end="", flush=True)
#         async for token in client.stream(
#             provider = "openai",
#             model    = "gpt-4o-mini",
#             prompt   = "Count 1 to 5 slowly.",
#             max_tokens = 50,
#         ):
#             print(token, end="", flush=True)
#         print()
#         print("  STATUS : PASSED")
#     except Exception as e:
#         print(f"\n  STATUS : FAILED — {e}")
#     print()
#
#     # ── STEP 7: Test Ollama streaming ────────────────────────────
#     print("=" * 50)
#     print("STEP 7 — Ollama stream()")
#     print("=" * 50)
#     try:
#         print("  Tokens : ", end="", flush=True)
#         async for token in client.stream(
#             provider = "ollama",
#             model    = "llama3",
#             prompt   = "Count 1 to 5 slowly.",
#         ):
#             print(token, end="", flush=True)
#         print()
#         print("  STATUS : PASSED")
#     except Exception as e:
#         print(f"\n  STATUS : FAILED — {e}")
#     print()
#
#     # ── FINAL SUMMARY ─────────────────────────────────────────────
#     print("=" * 50)
#     print("ALL TESTS COMPLETE")
#     print("=" * 50)
#
# asyncio.run(main())
