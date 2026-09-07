"""Canonical entry point for the recurrent-QDQN ablation protocol.

The primary successor remains ``run_native_qa2c_successor.py``. This wrapper
keeps historical QDQN invocation compatible while exposing its ablation role
in the filename used for new experiments.
"""

from qbbr.scripts.run_full_successor_qrl_bbr import main


if __name__ == "__main__":
    main()
