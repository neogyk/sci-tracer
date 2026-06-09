from typing import Final


def final_remainder(a_mod: int, floor_tail: int, modulus: int = 1000) -> int:
    return (a_mod + floor_tail) % modulus


def main() -> None:
    A100_mod_1000: Final[int] = 669
    floor_R100: Final[int] = 0
    modulus: Final[int] = 1000

    answer: int = final_remainder(A100_mod_1000, floor_R100, modulus)
    print(answer)


if __name__ == "__main__":
    main()