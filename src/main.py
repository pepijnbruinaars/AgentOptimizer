from CustomEnvironment.custom_environment import CustomEnvironment
from CustomEnvironment.custom_environment.env.custom_environment import (
    SimulationParameters,
)
from config import config
from preprocessing.load_data import load_data


def main() -> None:
    # Load data and show the first 20 rows for inspection
    data = load_data(config)
    print(data.head(20))
    simulation_parameters: SimulationParameters = {
        "start_timestamp": data["start_timestamp"].min()
    }

    # Create an instance of the custom environment
    env = CustomEnvironment(
        data,
        simulation_parameters,
    )

    # Perform a step in the environment
    for i in range(10):
        env.step({})


if __name__ == "__main__":
    main()
