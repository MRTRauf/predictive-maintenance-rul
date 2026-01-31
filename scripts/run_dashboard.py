import subprocess
import sys
from pathlib import Path


def main() -> None:
    dashboard_path = Path("src") / "pm_rul" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        print("Run this command:")
        print("python -m streamlit run src/pm_rul/dashboard.py")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
