"""Reject empty/all-skipped CI reports and optionally require every case to run."""

import argparse
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--no-skips", action="store_true")
    args = parser.parse_args()
    cases = ET.parse(args.report).findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    )
    print(
        f"CI report: {len(cases)} collected, {len(cases) - skipped} executed, {skipped} skipped, {failed} failed"
    )
    if not cases or skipped == len(cases) or failed or (args.no_skips and skipped):
        raise SystemExit("CI report does not satisfy the required execution policy")


if __name__ == "__main__":
    main()
