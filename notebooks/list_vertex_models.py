"""
List models available through Vertex AI for this project/location.

Usage:
    python notebooks/list_vertex_models.py

Optional environment:
    GOOGLE_CLOUD_PROJECT=...
    GOOGLE_CLOUD_LOCATION=...
    VERTEX_LOCATION=...
"""

import argparse
import os
import sys

from google import genai

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_PROJECT = "project-89fb5bc1-75d8-4158-896"
DEFAULT_LOCATION = "us-central1"


def value_or_default(value, default):
    return value if value else default


def model_attr(model, name, default=""):
    value = getattr(model, name, default)
    return value if value is not None else default


def main():
    parser = argparse.ArgumentParser(description="List Vertex AI models visible to google-genai.")
    parser.add_argument(
        "--project",
        default=value_or_default(os.environ.get("GOOGLE_CLOUD_PROJECT"), DEFAULT_PROJECT),
    )
    parser.add_argument(
        "--location",
        default=value_or_default(
            os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEX_LOCATION"),
            DEFAULT_LOCATION,
        ),
    )
    args = parser.parse_args()

    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
    )

    print(f"Vertex project : {args.project}")
    print(f"Vertex location: {args.location}")
    print()

    count = 0
    for model in client.models.list():
        count += 1
        name = model_attr(model, "name")
        display_name = model_attr(model, "display_name")
        version = model_attr(model, "version")
        supported_actions = model_attr(model, "supported_actions", [])
        actions = ", ".join(supported_actions) if supported_actions else ""

        print(name)
        if display_name:
            print(f"  display_name: {display_name}")
        if version:
            print(f"  version     : {version}")
        if actions:
            print(f"  actions     : {actions}")
        print()

    print(f"Total models: {count}")


if __name__ == "__main__":
    main()
