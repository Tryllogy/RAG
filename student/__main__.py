import fire
from src import fire_launch
from pydantic import ValidationError


if __name__ == "__main__":
    try:
        fire.Fire(fire_launch.FireLaunch)
    except ValidationError as e:
        for error in e.errors():
            print(f"Error in field '{error['loc'][0]}': {error['msg']}")
