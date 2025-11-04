#!/usr/bin/env python3
"""
utils_llm.py – Inspect LLM API capabilities for your API key.

Features:
- Robust .env loading (next to script and project root)
- List models (OpenAI/Groq) or tags (Ollama). Azure shows a note (deployments).
- Rate limit test for the configured provider/model
- Deep probe (OpenAI/Groq): tries a set of known model IDs with tiny requests to
  surface which ones are accessible and what rate-limit headers they return.

Usage examples:
  python utils_llm.py --provider openai
  python utils_llm.py --provider openai --rate-test
  python utils_llm.py --provider openai --deep
  python utils_llm.py --provider groq --deep
  python utils_llm.py --provider openai --json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv


# ---------- .env robust laden ----------
def _load_env() -> None:
    # .env neben dem Skript
    env_here = Path(__file__).with_name(".env")
    # .env eine Ebene höher (Projektroot)
    env_root = Path(__file__).resolve().parent.parent / ".env"

    loaded = False
    if env_here.exists():
        load_dotenv(dotenv_path=env_here, override=True)
        loaded = True
    if env_root.exists():
        load_dotenv(dotenv_path=env_root, override=True)
        loaded = True
    if not loaded:
        load_dotenv()  # Fallback: CWD


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise EnvironmentError(f"Missing environment variable: {key}")
    return val.strip()


# ---------- Listing ----------
def list_models(provider: str) -> Dict[str, Any]:
    p = provider.lower()
    if p == "openai":
        url = "https://api.openai.com/v1/models"
        key = _require_env("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {key}"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    if p == "groq":
        url = "https://api.groq.com/openai/v1/models"
        key = _require_env("GROQ_API_KEY")
        headers = {"Authorization": f"Bearer {key}"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    if p == "azure":
        return {
            "note": (
                "Azure OpenAI listet keine Modelle via Runtime-API. "
                "Verwende das Azure Portal oder die Cognitive Services Management API, "
                "um Deployments aufzulisten."
            )
        }

    if p == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    raise ValueError(f"Unknown provider: {provider}")


# ---------- Rate-Limit-Test ----------
def _rl_headers(resp: requests.Response) -> Dict[str, str]:
    return {
        k: v for k, v in resp.headers.items()
        if k.lower().startswith("x-ratelimit") or k.lower() in ("retry-after", "x-request-id")
    }


def rate_limit_test(provider: str, model: str = None) -> Dict[str, Any]:
    p = provider.lower()
    out: Dict[str, Any] = {"provider": provider}

    try:
        if p in ("openai", "groq", "ollama"):
            if p == "openai":
                base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                key = _require_env("OPENAI_API_KEY")
                default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            elif p == "groq":
                base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
                key = _require_env("GROQ_API_KEY")
                default_model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            else:  # ollama
                base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
                default_model = os.getenv("OLLAMA_MODEL", "llama3.1")
                headers = {"Content-Type": "application/json"}
                if os.getenv("OLLAMA_API_KEY"):
                    headers["Authorization"] = f"Bearer {os.getenv('OLLAMA_API_KEY')}"

            url = f"{base}/chat/completions"
            mdl = model or default_model
            payload = {
                "model": mdl,
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0,
                "max_tokens": 1,
                "top_p": 1,
                "stream": False,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            out["status"] = resp.status_code
            out["rate_limit_headers"] = _rl_headers(resp)
            try:
                out["body"] = resp.json()
            except Exception:
                out["body"] = resp.text
            return out

        if p == "azure":
            base = _require_env("AZURE_OPENAI_BASE_URL").rstrip("/")
            key = _require_env("AZURE_OPENAI_API_KEY")
            deployment = model or _require_env("AZURE_OPENAI_DEPLOYMENT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            url = f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": key, "Content-Type": "application/json"}
            payload = {"messages": [{"role": "user", "content": "ping"}], "temperature": 0, "max_tokens": 1}
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            out["status"] = resp.status_code
            out["rate_limit_headers"] = _rl_headers(resp)
            try:
                out["body"] = resp.json()
            except Exception:
                out["body"] = resp.text
            return out

        raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        out["error"] = str(e)
        return out


# ---------- Deep Probe: testet mehrere Modell-IDs ----------
def deep_probe(provider: str, candidates: List[str]) -> List[Dict[str, Any]]:
    p = provider.lower()
    results: List[Dict[str, Any]] = []

    if p not in ("openai", "groq"):
        return [{"note": "Deep probe currently implemented for openai/groq only."}]

    if p == "openai":
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = _require_env("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    else:  # groq
        base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        key = _require_env("GROQ_API_KEY")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    url = f"{base}/chat/completions"

    for mdl in candidates:
        payload = {
            "model": mdl,
            "messages": [{"role": "user", "content": "probe"}],
            "temperature": 0,
            "max_tokens": 1,
            "top_p": 1,
            "stream": False,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            item = {
                "model": mdl,
                "status": resp.status_code,
                "rate_limit_headers": _rl_headers(resp)
            }
            try:
                item["body"] = resp.json()
            except Exception:
                item["body"] = resp.text
            results.append(item)
        except Exception as e:
            results.append({"model": mdl, "error": str(e)})

    return results


def main():
    _load_env()

    parser = argparse.ArgumentParser(description="LLM Info CLI")
    parser.add_argument("--provider", "-p",
                        default=os.getenv("LLM_PROVIDER", "openai"),
                        help="openai | groq | azure | ollama")
    parser.add_argument("--rate-test", action="store_true",
                        help="Perform a tiny chat request and show X-RateLimit headers")
    parser.add_argument("--model", help="Override model/deployment for rate-test")
    parser.add_argument("--deep", action="store_true",
                        help="Probe several known model IDs (openai/groq) to surface access and headers")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    provider = args.provider.strip().lower()

    # 1) Modelle/Deployments
    models_info = list_models(provider)

    # 2) Optional: Rate-Limit-Test
    rl_info = rate_limit_test(provider, model=args.model) if args.rate_test else None

    # 3) Optional: Deep-Probe-Kandidaten
    deep_info = None
    if args.deep:
        if provider == "openai":
            candidates = [
                # Häufige OpenAI-Modelle (nur Beispiele; nicht alle sind jedem Key freigeschaltet)
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-4.1",
                "o4-mini",
                "o4",
            ]
        elif provider == "groq":
            candidates = [
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "llama-3.2-11b-text-preview",
                "llama-3.2-90b-text-preview",
            ]
        else:
            candidates = []

        deep_info = deep_probe(provider, candidates) if candidates else [{"note": "No candidates for this provider."}]

    if args.json:
        out = {
            "provider": provider,
            "models": models_info,
            "rate_limit_test": rl_info,
            "deep_probe": deep_info,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    # Menschlich formatiert
    print(f"\n=== Provider: {provider} ===")

    print("--- Models / Deployments ---")
    if provider in ("openai", "groq"):
        if isinstance(models_info, dict) and "data" in models_info:
            for m in models_info["data"]:
                print(" •", m.get("id"))
        else:
            print(models_info)
    elif provider == "ollama":
        tags = models_info.get("models") or models_info.get("data") or models_info
        if isinstance(tags, list):
            for m in tags:
                mid = m.get("name") or m.get("model") or m
                print(" •", mid)
        else:
            print(models_info)
    else:
        print(models_info)

    if rl_info is not None:
        print("\n--- Rate Limit Test ---")
        print("Status:", rl_info.get("status"))
        rl_headers = rl_info.get("rate_limit_headers") or {}
        if rl_headers:
            for k, v in rl_headers.items():
                print(f"{k}: {v}")
        else:
            print("No rate-limit headers returned.")
        if "error" in rl_info:
            print("Error:", rl_info["error"])

    if deep_info is not None:
        print("\n--- Deep Probe (selected models) ---")
        for item in deep_info:
            if "model" in item:
                line = f"{item['model']}: {item.get('status', 'n/a')}"
                print(line)
                headers = item.get("rate_limit_headers") or {}
                for k, v in headers.items():
                    print(f"   {k}: {v}")
            else:
                print(item)

    print("\nDone.\n")


if __name__ == "__main__":
    try:
        main()
    except EnvironmentError as e:
        print(f"Environment error: {e}", file=sys.stderr)
        sys.exit(2)
    except requests.HTTPError as e:
        resp = e.response
        print(f"HTTP error: {e}", file=sys.stderr)
        if resp is not None:
            try:
                print(resp.status_code, resp.json(), file=sys.stderr)
            except Exception:
                print(resp.status_code, resp.text, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unhandled error: {e}", file=sys.stderr)
        sys.exit(1)