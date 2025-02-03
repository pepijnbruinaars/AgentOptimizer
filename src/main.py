from CustomEnvironment.custom_environment import CustomEnvironment
from config import config
from preprocessing.load_data import load_data


def main() -> None:
    # Load data and show the first 20 rows for inspection
    data = load_data(config)
    print(data.head(20))

    # Create an instance of the custom environment
    env = CustomEnvironment()


if __name__ == "__main__":
    main()
