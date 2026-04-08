"""
email_check.py — Paste any email text and instantly get it classified
Usage:
    python email_check.py                  # interactive paste mode
    python email_check.py -f email.txt     # classify from a .txt file
    python email_check.py -t "your text"   # classify inline text
"""

import os
import sys
import json
import argparse

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.prompt import Prompt
except ImportError:
    os.system("pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.prompt import Prompt

try:
    from openai import OpenAI
except ImportError:
    os.system("pip install openai -q")
    from openai import OpenAI

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
# Try to use OpenEnv-provided credentials first, then fall back for standalone use
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = """You are an expert email security and triage assistant.

Classify the given email into ONE of these categories:

- spam       : Unwanted bulk mail, unsolicited promotions, newsletters you didn't sign up for
- scam       : Phishing, fake prizes, fraud, impersonation, fake invoices, advance fee fraud
- important  : Requires action — deadlines, client messages, security alerts, approvals, urgent work
- normal     : Routine, informational, expected correspondence, automated digests, FYI updates

Also provide:
- confidence : high / medium / low
- reason     : one sentence explaining why
- warning    : (only if scam/spam) specific red flags found

Respond ONLY with a valid JSON object, no markdown, no extra text:
{
  "category": "scam",
  "confidence": "high",
  "reason": "Claims you won a lottery you never entered and asks for bank details.",
  "warning": "Fake prize, requests personal banking info, urgent pressure tactics"
}"""

# ── Category styles ───────────────────────────────────────────────────────────
STYLES = {
    "spam":      ("🚫 SPAM",       "bold white on red",         "red"),
    "scam":      ("☠️  SCAM",       "bold white on dark_red",    "dark_red"),
    "important": ("⭐ IMPORTANT",  "bold white on dark_orange", "dark_orange"),
    "normal":    ("✅ SAFE/NORMAL","bold white on green",       "green"),
}

CONFIDENCE_STYLE = {
    "high":   "bold green",
    "medium": "bold yellow",
    "low":    "bold red",
}

# ── Classifier ────────────────────────────────────────────────────────────────
def classify(email_text: str) -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this email:\n\n{email_text}"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        console.print(f"[bold red]Classification error: {e}[/]")
        return {"category": "normal", "confidence": "low", "reason": "Could not classify.", "warning": ""}

# ── Display result ────────────────────────────────────────────────────────────
def show_result(email_text: str, result: dict):
    category   = result.get("category", "normal").lower()
    confidence = result.get("confidence", "medium").lower()
    reason     = result.get("reason", "")
    warning    = result.get("warning", "")

    label, badge_style, border_color = STYLES.get(category, STYLES["normal"])

    # Preview of email
    preview = email_text.strip()[:200]
    if len(email_text) > 200:
        preview += "..."

    console.print()
    console.print(Rule("[bold white] CLASSIFICATION RESULT [/]", style="white"))
    console.print()

    # Email preview panel
    console.print(Panel(
        Text(preview, style="dim white"),
        title="[bold]📨 Email Preview[/]",
        border_style="bright_black",
        padding=(0, 2),
    ))

    console.print()

    # Verdict
    verdict_text = Text()
    verdict_text.append("\n  VERDICT:     ", style="dim")
    verdict_text.append(f"  {label}  ", style=badge_style)
    verdict_text.append("\n\n  CONFIDENCE:  ", style="dim")
    verdict_text.append(confidence.upper(), style=CONFIDENCE_STYLE.get(confidence, "white"))
    verdict_text.append("\n\n  REASON:      ", style="dim")
    verdict_text.append(reason, style="white")

    if warning and category in ("spam", "scam"):
        verdict_text.append("\n\n  ⚠️  WARNING:   ", style="bold red")
        verdict_text.append(warning, style="bold yellow")

    verdict_text.append("\n")

    console.print(Panel(
        verdict_text,
        title=f"[bold {border_color}]🔍 Analysis[/]",
        border_style=border_color,
        padding=(0, 1),
    ))

    # Advice
    advice = {
        "scam":      "🛑 [bold red]DO NOT click any links, reply, or provide personal information. Delete this email.[/]",
        "spam":      "🗑️  [yellow]This is unwanted mail. Safe to delete or mark as spam.[/]",
        "important": "📌 [bold green]This email needs your attention. Reply or take action.[/]",
        "normal":    "👍 [green]This looks like a normal safe email.[/]",
    }
    console.print(f"\n  {advice.get(category, '')}\n")

# ── Input modes ───────────────────────────────────────────────────────────────
def get_email_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_email_interactive() -> str:
    console.print(Panel(
        "[dim]Paste your email text below.\n"
        "When done, type [bold cyan]END[/bold cyan] on a new line and press Enter.[/dim]",
        title="[bold yellow]📧 Email Classifier[/]",
        border_style="yellow",
    ))
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Classify any email as spam/scam/important/normal")
    parser.add_argument("-f", "--file", help="Path to a .txt file containing the email")
    parser.add_argument("-t", "--text", help="Email text directly as a string")
    args = parser.parse_args()

    if not API_KEY:
        console.print(Panel(
            "[bold red]No API key found!\n\n[white]"
            "Set your token first:\n"
            "  [cyan]$env:HF_TOKEN='hf_xxxxxxxxxxxx'[/cyan]\n\n"
            "Get it at: https://huggingface.co/settings/tokens",
            title="❌ Error", border_style="red"
        ))
        sys.exit(1)

    # Get email text
    if args.file:
        console.print(f"[dim]Reading from file: {args.file}[/]")
        email_text = get_email_from_file(args.file)
    elif args.text:
        email_text = args.text
    else:
        email_text = get_email_interactive()

    if not email_text.strip():
        console.print("[bold red]No email text provided. Exiting.[/]")
        sys.exit(1)

    # Classify
    with console.status("[bold yellow]Analysing email...[/]", spinner="dots"):
        result = classify(email_text)

    show_result(email_text, result)

    # Loop for more
    while True:
        again = input("\n  Check another email? (y/n): ").strip().lower()
        if again != "y":
            break
        email_text = get_email_interactive()
        if email_text.strip():
            with console.status("[bold yellow]Analysing email...[/]", spinner="dots"):
                result = classify(email_text)
            show_result(email_text, result)

    console.print("\n[dim]Goodbye! Stay safe from scams. 👋[/]\n")

if __name__ == "__main__":
    main()
