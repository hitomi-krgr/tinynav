# Default planner entrypoint. Re-exports a concrete planner implementation;
# swap the import below to select centerline / turn / vanilla.
from tinynav.core.planning_node_turn import PlanningNode, main

__all__ = ["PlanningNode", "main"]

if __name__ == '__main__':
    main()
