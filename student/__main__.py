import fire


def test(x="salut"):
    return f"{x} Feurlogy!"


def main() -> None:
    fire.Fire(test)


if __name__ == "__main__":
    main()
