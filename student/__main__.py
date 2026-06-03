import fire
from src import fire_launch
from pydantic import ValidationError
import sys


if __name__ == "__main__":
    try:
        fire.Fire(fire_launch.FireLaunch)
    except ValidationError as e:
        print("Validation Error:", file=sys.stderr)
        for error in e.errors():
            print(f"  Field '{error['loc'][0]}': {error['msg']}",
                  file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File Not Found: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Missing Key in Data: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
