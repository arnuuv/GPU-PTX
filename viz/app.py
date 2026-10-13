"""Local viewer for IR, PTX, simulator state, and analytical estimates.

Run ``python -m ptxlab dashboard`` or ``streamlit run viz/app.py``.
Requires the optional ``viz`` extra. Figures labelled as estimates are model
output, not GPU measurements.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ptxlab.report import report_from_path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(path for path in (ROOT / "examples").glob("*.py"))


def main() -> None:
    st.set_page_config(page_title="PTXLab", layout="wide")
    st.title("PTXLab")
    st.caption("Functional simulator output and an analytical timing model. Cycle counts are not GPU measurements.")
    names = [path.stem for path in EXAMPLES]
    choice = st.selectbox("Kernel", names, index=names.index("vector_add") if "vector_add" in names else 0)
    report = report_from_path(ROOT / "examples" / f"{choice}.py")
    cols = st.columns(4)
    cols[0].metric("Instructions", report["instructions"])
    cols[1].metric("Divergent branches", report["divergent_branches"])
    cols[2].metric("Global transactions", report["memory_transactions"])
    cycles = report["estimated_cycles"]
    cols[3].metric("Estimated cycles", "n/a" if cycles is None else cycles)
    st.write(f"Occupancy {report['occupancy']:.3f}. Max abs error by output: {report['errors']}.")
    left, right = st.columns(2)
    left.subheader("IR")
    left.code(report["ir"], language="text")
    right.subheader("PTX")
    right.code(report["ptx"], language="text")
    st.subheader("Instruction counts")
    st.dataframe(
        [{"op": op, "count": count} for op, count in sorted(report["inst_counts"].items())],
        hide_index=True,
    )
    st.subheader("Memory access timeline")
    st.dataframe(report["timeline"], hide_index=True)
    st.subheader("Warp trace (first instructions)")
    st.dataframe(report["trace"], hide_index=True)
    st.subheader("Model assumptions")
    for line in report["assumptions"]:
        st.text(line)


main()
