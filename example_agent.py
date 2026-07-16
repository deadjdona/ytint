"""Example showing how to load agents configuration."""

from config.agents import load_agents, get_agent
import json


def main():
    agents = load_agents()
    print("Loaded agents:")
    print(json.dumps(agents, indent=2))

    a = get_agent("default")
    print("\nDefault agent config:")
    print(json.dumps(a, indent=2))


if __name__ == "__main__":
    main()
