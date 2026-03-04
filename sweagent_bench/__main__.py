"""Allow ``python -m sweagent_bench`` to run the CLI."""
from sweagent_bench.main import main
import sys

sys.exit(main())
