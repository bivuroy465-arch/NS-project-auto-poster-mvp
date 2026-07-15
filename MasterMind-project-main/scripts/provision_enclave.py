#!/usr/bin/env python
"""Enclave provisioning helper — run once to encrypt your API keys.

Usage
-----
    # 1. Generate a master key (do this once; store in ENCLAVE_MASTER_KEY secret):
    python scripts/provision_enclave.py --generate-key

    # 2. Encrypt all API keys found in your current .env file:
    #    Reads ENCLAVE_MASTER_KEY from the environment and encrypts every
    #    KEY=value pair whose name appears in CREDENTIAL_NAMES.
    python scripts/provision_enclave.py --encrypt-all

    # 3. Encrypt a single credential interactively:
    python scripts/provision_enclave.py --encrypt --name OPENAI_API_KEY

Output lines are in the format:
    ENCRYPTED_<NAME>=<name>:<nonce_b64>:<ciphertext_b64>

Paste these as CI/CD secrets (GitHub → Settings → Secrets, or Vercel → Vars).

Security note
-------------
This script reads the plaintext secret from stdin (when --encrypt is used)
or from os.environ (when --encrypt-all is used) and immediately encrypts it.
The plaintext is never written to disk, logged, or sent over the network.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import os
import sys

# Ensure the project root is on sys.path so we can import src.enclave.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enclave.enclave import encrypt_credential  # noqa: E402

_CREDENTIAL_NAMES = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "STABILITY_API_KEY",
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_SECRET",
    "LINKEDIN_ACCESS_TOKEN",
    "LINKEDIN_AUTHOR_URN",
    "FACEBOOK_PAGE_ID",
    "FACEBOOK_PAGE_ACCESS_TOKEN",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
]


def cmd_generate_key() -> None:
    key_bytes = os.urandom(32)
    b64 = base64.urlsafe_b64encode(key_bytes).decode()
    print("\n--- Generated ENCLAVE_MASTER_KEY ---")
    print(f"ENCLAVE_MASTER_KEY={b64}")
    print("\nAdd this to your CI/CD secrets as ENCLAVE_MASTER_KEY.")
    print("NEVER commit it to source control.\n")


def _load_master_key() -> bytes:
    raw = os.environ.get("ENCLAVE_MASTER_KEY", "")
    if not raw:
        print("ERROR: ENCLAVE_MASTER_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return base64.urlsafe_b64decode(raw)


def cmd_encrypt_one(name: str) -> None:
    master_key = _load_master_key()
    plaintext = getpass.getpass(f"Enter plaintext value for {name}: ")
    if not plaintext:
        print("Empty value — skipping.", file=sys.stderr)
        return
    cred = encrypt_credential(name, plaintext, master_key)
    print(f"\nENCRYPTED_{name}={cred.to_env_value()}")


def cmd_encrypt_all() -> None:
    master_key = _load_master_key()
    encrypted: list[str] = []
    for name in _CREDENTIAL_NAMES:
        value = os.environ.get(name, "").strip()
        if not value:
            print(f"  SKIP  {name} (not set in environment)", file=sys.stderr)
            continue
        cred = encrypt_credential(name, value, master_key)
        line = f"ENCRYPTED_{name}={cred.to_env_value()}"
        encrypted.append(line)
        print(f"  OK    {name}")

    if encrypted:
        print("\n--- Copy these to your CI/CD secrets ---")
        for line in encrypted:
            print(line)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enclave provisioning helper — encrypt API keys for the Credential Enclave."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate a new random 256-bit master key.",
    )
    group.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt a single credential (prompted interactively).",
    )
    group.add_argument(
        "--encrypt-all",
        action="store_true",
        help="Encrypt all credentials found in the current environment.",
    )
    parser.add_argument(
        "--name",
        help="Credential name (required with --encrypt).",
    )
    args = parser.parse_args()

    if args.generate_key:
        cmd_generate_key()
    elif args.encrypt:
        if not args.name:
            parser.error("--encrypt requires --name <CREDENTIAL_NAME>")
        cmd_encrypt_one(args.name)
    elif args.encrypt_all:
        cmd_encrypt_all()


if __name__ == "__main__":
    main()
