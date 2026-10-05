"""Check the exact release inputs; this command never uploads artifacts."""

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    version = project["version"]
    require(
        not args.tag or args.tag == f"v{version}", "tag does not match project version"
    )
    locked = [
        p
        for p in tomllib.loads(Path("uv.lock").read_text())["package"]
        if p["name"] == "torchrir"
    ]
    require(
        len(locked) == 1 and locked[0]["version"] == version,
        "lock version does not match project",
    )
    wheels = list(args.dist.glob("*.whl"))
    sdists = list(args.dist.glob("*.tar.gz"))
    require(
        len(wheels) == 1 and wheels[0].name == f"torchrir-{version}-py3-none-any.whl",
        "wheel filename/count mismatch",
    )
    require(
        len(sdists) == 1 and sdists[0].name == f"torchrir-{version}.tar.gz",
        "sdist filename/count mismatch",
    )
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        require(len(names) == 1, "wheel must contain one metadata file")
        metadata = BytesParser().parsebytes(wheel.read(names[0]))
        require(
            metadata["Name"] == "torchrir" and metadata["Version"] == version,
            "wheel metadata version/name mismatch",
        )
        require(
            metadata["License-Expression"] == "Apache-2.0", "wheel license mismatch"
        )
    with tarfile.open(sdists[0], "r:gz") as sdist:
        roots = [n for n in sdist.getnames() if len(PurePosixPath(n).parts) == 2]
        require(f"torchrir-{version}/CHANGELOG.md" in roots, "sdist changelog missing")
        names = [n for n in roots if PurePosixPath(n).name == "PKG-INFO"]
        require(len(names) == 1, "sdist must contain one root metadata file")
        stream = sdist.extractfile(names[0])
        if stream is None:
            raise ValueError("sdist metadata missing")
        metadata = BytesParser().parsebytes(stream.read())
        require(
            metadata["Name"] == "torchrir" and metadata["Version"] == version,
            "sdist metadata version/name mismatch",
        )
    print(f"Validated torchrir {version}: tag, lock, wheel, sdist, license, changelog")


if __name__ == "__main__":
    main()
