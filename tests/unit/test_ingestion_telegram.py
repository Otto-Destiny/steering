from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.ingestion.telegram import telegram_sources


def test_telegram_json_export_extracts_plain_and_structured_links(tmp_path: Path) -> None:
    export = tmp_path / "result.json"
    export.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "text": ["Paper: ", {"type": "link", "text": "https://arxiv.org/abs/1."}],
                        "text_entities": [],
                    },
                    {
                        "text": "Repository",
                        "text_entities": [
                            {
                                "type": "text_link",
                                "text": "source code",
                                "href": "https://github.com/example/tool",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert telegram_sources(export) == [
        "https://arxiv.org/abs/1",
        "https://github.com/example/tool",
    ]


def test_telegram_html_export_uses_anchor_target_not_only_visible_text(tmp_path: Path) -> None:
    export = tmp_path / "messages.html"
    export.write_text(
        """
        <html><body>
          <div class="message">Direct https://openreview.net/forum?id=abc</div>
          <div class="message">Project <a href="https://github.com/example/project">github.com/...</a></div>
          <div class="service">https://example.org/not-a-message</div>
        </body></html>
        """,
        encoding="utf-8",
    )

    assert telegram_sources(export) == [
        "https://openreview.net/forum?id=abc",
        "https://github.com/example/project",
    ]


def test_telegram_import_rejects_wrong_shapes_and_formats(tmp_path: Path) -> None:
    wrong_shape = tmp_path / "array.json"
    wrong_shape.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        telegram_sources(wrong_shape)

    unsupported = tmp_path / "messages.txt"
    unsupported.write_text("https://example.org", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON or HTML"):
        telegram_sources(unsupported)
