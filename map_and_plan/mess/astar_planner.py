"""Methods for planning. Our system assumes this is given and fixed.
"""

from collections import defaultdict
import numpy as np
import heapq


class Astar_Planner:
    """A* planner.
    """
    def __init__(self, model, action_list, max_depth, max_node_expansions,
                 goal_check, heuristic, state_to_tuple):
        self.model = model
        self.action_list = action_list
        self.max_depth = max_depth
        self.max_node_expansions = max_node_expansions
        self.goal_check = goal_check
        self.heuristic = heuristic
        self.rng = np.random.RandomState(0)
        self.state_to_tuple = state_to_tuple

    def get_action_list(self):
        """Get action list.
        """
        return self.action_list

    def update_action_list(self, action_list):
        """Update action list.
        """
        self.action_list = action_list

    def plan(self, init_state, goal):
        """A* search. Returns a PlannerOutput object.
        """
        pqueue = PriorityQueue()
        visited = set()
        root = SearchTreeEntry(init_state,
                               state_sequence=[],
                               action_sequence=[],
                               abstract_state_sequence=[],
                               abstract_action_sequence=[],
                               cost=0,
                               cost_sequence=[0],
                               depth=0)
        pqueue.push(root, self.heuristic(init_state, goal))
        nodes_expanded = 0
        while (not pqueue.is_empty() and
               nodes_expanded < self.max_node_expansions):
            entry = pqueue.pop()
            if self.goal_check(entry.state, goal):
                entry.state_sequence.append(entry.state)
                entry.abstract_state_sequence.append(entry.state)
                # print("expanded {} nodes".format(nodes_expanded))
                #cost_to_go = (entry.cost_sequence[-1]-
                #              np.array(entry.cost_sequence))
                #print(cost_to_go)
                #cost_to_go_pred = [self.heuristic(s, goal) for s
                #                   in entry.state_sequence]
                #if any(cost_to_go_pred > cost_to_go):
                #    raise Exception("Heuristic is not admissible")
                return PlannerOutput(entry.action_sequence, entry.cost,
                                     entry.state_sequence, nodes_expanded,
                                     entry.abstract_state_sequence,
                                     entry.abstract_action_sequence)
            if not self._in_set(entry.state, visited):
                self._add_to_set(entry.state, visited)
                if entry.depth == self.max_depth:
                    continue
                nodes_expanded += 1
                for action in self.action_list:
                    next_state_lst, action_lst = self.model(entry.state, action)
                    if not next_state_lst:
                        # Action was a Policy which only noop'ed.
                        continue
                    cost = self._action_cost(action)
                    new_entry = SearchTreeEntry(
                        next_state_lst[-1],
                        entry.state_sequence+[entry.state]+next_state_lst,
                        entry.action_sequence+[action_lst],
                        entry.abstract_state_sequence+[entry.state],
                        entry.abstract_action_sequence+[action],
                        entry.cost+cost,
                        entry.cost_sequence+[entry.cost+cost for _
                                             in range(len(next_state_lst))],
                        entry.depth+1)
                    pqueue.push(new_entry, (entry.cost+cost+
                                            self.heuristic(next_state_lst, goal)))


    def _add_to_set(self, state, state_set):
        state_tuple = self.state_to_tuple(state)
        state_set.add(state_tuple)

    def _in_set(self, state, state_set):
        state_tuple = self.state_to_tuple(state)
        return state_tuple in state_set

    @staticmethod
    def _action_cost(_action):
        return 1

class SearchTreeEntry:
    """Entry in search tree for A*.
    """
    def __init__(self, state, state_sequence, action_sequence,
                 abstract_state_sequence, abstract_action_sequence,
                 cost, cost_sequence, depth):
        self.state = state
        self.state_sequence = state_sequence
        self.action_sequence = action_sequence
        self.abstract_state_sequence = abstract_state_sequence
        self.abstract_action_sequence = abstract_action_sequence
        self.cost = cost
        self.cost_sequence = cost_sequence
        self.depth = depth


class PlannerOutput:
    """Output of planner. If the plan has length n, then state_traj has
    length n+1. state_traj[i] is the state before running the action
    given by plan[i].
    """
    def __init__(self, plan, cost, state_traj, nodes_expanded,
                 abstract_state_traj, abstract_plan):
        self.plan = plan
        self.cost = cost
        self.state_traj = state_traj
        self.nodes_expanded = nodes_expanded
        self.abstract_state_traj = abstract_state_traj
        self.abstract_plan = abstract_plan

    def resource_val(self):
        """Get value of resource that the ZPD is defined with respect to.
        """
        if settings.CAP_HORIZON:
            return len(self.abstract_plan)
        if settings.CAP_NODE_EXPANSIONS:
            return self.nodes_expanded
        raise Exception("Should never get here")

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        "Enqueue the 'item' into the queue"
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        "Pop from the queue"
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        "Returns true if the queue is empty"
        return len(self.heap) == 0
