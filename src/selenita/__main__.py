"""Enables ``python -m selenita --acp`` (and the integration tests that launch it)."""

from selenita.cli.main import main

raise SystemExit(main())
