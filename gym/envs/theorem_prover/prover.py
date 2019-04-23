import pickle
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from copy import deepcopy
from Inequality.logic.logic import Entity, LogicStatement


nodenames = ["Real",    "NonNegative",    "BiggerOrEqual",    "SmallerOrEqual",    "Equivalent",
             "add",    "opp",    "sub",    "mul",    "sqr",    "sqrt",    "inv",    "geometric_mean",    "identity"] \
            + ["input{}".format(index) for index in range(1, 6)] \
            + ["{}".format(i) for i in range(-10, 11)] \
            + ["NOOP"]

nodename2index = {value: index for index, value in enumerate(nodenames)}

configuration = {
    "proof_dir": "/home/albert/Albert/MyProjects/can18/ineqSolver/Inequality/data/standard_theorem_dataset",
    "max_theorems": 25,
    "max_entities": 100,
    "max_ground_truth": 50,
    "max_objectives": 1,
    "max_operands": 5,
    "max_edges": 200,
    "max_nodes": 50,
    "max_node_types": 40,
    "max_configuration": 1000
}


def initialize_proof(proof_dir, np_random, total_configuration=1000):
    configuration_index = int(np_random.randint(total_configuration, size=1))
    return pickle.load(open(proof_dir + "/proof_{}.p".format(configuration_index), "rb"))


def entity_to_graph(node_feat_l, node_ind, entity, source):
    if entity.rnc_operands is None:
        node_feat_l.append(nodename2index[entity.name])
        node_ind.append(entity.to_string())
        return [[len(node_ind), source], [source, len(node_ind)]]
    else:
        node_feat_l.append(nodename2index[entity.recent_numerical_function.name])
        node_ind.append(entity.to_string())
        new_source = len(node_ind)
        data = [[new_source, source], [source, new_source]]
        for e in entity.rnc_operands:
            data += entity_to_graph(node_feat_l, node_ind, e, new_source)
        return data


def final_entity_to_graph(entity):
    if entity.rnc_operands is None:
        return [nodename2index[entity.name]], [], [entity.name]
    root = nodename2index[entity.recent_numerical_function.name]
    node_feat_l = [root]
    node_ind = [entity.name]
    edges = list()
    for e in entity.rnc_operands:
        edges += entity_to_graph(node_feat_l, node_ind, e, 0)
    return node_feat_l, edges, node_ind


def final_logic_statement_to_graph(logic_statement):
    root = nodename2index[logic_statement.logic_function.to_string()]
    node_feat_l = [root]
    node_ind = []
    edges = list()
    for e in logic_statement.entities:
        edges += entity_to_graph(node_feat_l, node_ind, e, 0)
    return node_feat_l, edges, node_ind


class TheoremProver(gym.Env):
    """Theorem proving environment

    The observation is a 3 tuple of: the adjacency matrices of the entity graphs,
    the adjacency matrices of the ground truth logic statement graphs,
    and the adjacency matrices of the objective logic statement graph.

    The action is a tuple of: the index of the lemma chosen,
    and the indices of the lemma operands chosen.
    The first entity in any proof must be named NOOP.

    Reward scheme:
    proof completed: 10,
    proof proceeded: 1,
    otherwise: 0.
    """
    def __init__(self, env_config=configuration):
        self.env_config = env_config
        self.proof_dir = env_config["proof_dir"]
        self.max_theorems = env_config["max_theorems"]
        self.max_entities = env_config["max_entities"]
        self.max_operands = env_config["max_operands"]
        self.max_edges = env_config["max_edges"]
        self.max_nodes = env_config["max_nodes"]
        self.max_node_types = env_config["max_node_types"]
        self.max_entities = env_config["max_entities"]
        self.max_ground_truth = env_config["max_ground_truth"]
        self.max_objectives = env_config["max_objectives"]

        self.action_space = spaces.MultiDiscrete(
            [self.max_theorems] +
            [self.max_entities] * self.max_operands
        )
        self.observation_space = spaces.Tuple(
            (
                # 2 by 2E matrix marking all edges
                [spaces.MultiDiscrete([self.max_nodes, self.max_nodes])] * 2 * self.max_edges
                # 1 by N matrix denoting all index types
                + [spaces.MultiDiscrete([self.max_node_types] * self.max_nodes)]
            )
            * (self.max_entities+self.max_ground_truth+self.max_objectives)
        )

        # Seeding
        self.np_random = None
        self.seed()

        # Initialize the entity node dictionary
        self.node_entities = dict()
        self.node_entity_names = []

        # Start the first game
        self.proof = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        lemma = self.proof.lemmas[action[0]]
        input_entities = list()
        for input_entity_index in action[1:]:
            entity = self.proof.entities[input_entity_index]
            if entity.name == "NOOP":
                break
            else:
                input_entities.append(entity)

        if lemma.input_no != len(input_entities):
            # Operand size mismatch
            info_string = "REWARD_OPERAND_SIZE_MISMATCH"
        else:
            info_string = self.proof.apply_theorem(lemma, input_entities)
        reward, done = {
            "REWARD_PROOF_COMPLETE": (10.0, True),
            "REWARD_THEOREM_PROCEEDED": (1.0, False),
            "REWARD_ASSUMPTION_INVALID": (0.0, False),
            "REWARD_DUPLICATED_RESULTS": (0.0, False),
            "REWARD_OPERAND_SIZE_MISMATCH": (0.0, False),
        }[info_string]
        if reward > 0:
            results = lemma.execute_th(input_entities)
            for graph in results["Conclusions"]+results["ExtraEntities"]:
                self._traverse_graph(graph)
        return self._get_obs(), reward, done

    def choose_node(self, graph_index, node_index):
        # (graph_index, node_index) -> entity
        return self.node_entities[self.node_entity_names[graph_index][node_index]]

    def _get_obs(self):
        graphs = list()
        self.node_entity_names = list()
        for entity in self.proof.entities:
            nodes, edges, node_ind = final_entity_to_graph(entity=entity)
            graphs.append((nodes, edges))
            self.node_entity_names.append(node_ind)

        if len(graphs) < self.max_entities:
            graphs += [([], [])] * (self.max_entities-len(graphs))
        elif len(graphs) == self.max_entities:
            pass
        else:
            raise NotImplemented
        if len(self.node_entity_names) < self.max_entities:
            self.node_entity_names += [[]] * (self.max_entities-len(self.node_entity_names))
        elif len(self.node_entity_names) == self.max_entities:
            pass
        else:
            raise NotImplemented

        for gt in self.proof.ground_truth:
            nodes, edges, node_ind = final_logic_statement_to_graph(logic_statement=gt)
            graphs.append((nodes, edges))
            self.node_entity_names.append(node_ind)

        if len(graphs) < self.max_entities + self.max_ground_truth:
            graphs += [([], [])] * (self.max_entities + self.max_ground_truth-len(graphs))
        elif len(graphs) == self.max_entities + self.max_ground_truth:
            pass
        else:
            raise NotImplemented
        if len(self.node_entity_names) < self.max_entities + self.max_ground_truth:
            self.node_entity_names += [[]] * (self.max_entities + self.max_ground_truth-len(self.node_entity_names))
        elif len(self.node_entity_names) == self.max_entities + self.max_ground_truth:
            pass
        else:
            raise NotImplemented

        for obj in self.proof.objectives:
            nodes, edges, node_ind = final_logic_statement_to_graph(logic_statement=obj)
            graphs.append((nodes, edges))
            self.node_entity_names.append(node_ind)

        if len(graphs) < self.max_entities + self.max_ground_truth + self.max_objectives:
            graphs += [([], [])] * (self.max_entities + self.max_ground_truth + self.max_objectives-len(graphs))
        elif len(graphs) == self.max_entities + self.max_ground_truth + self.max_objectives:
            pass
        else:
            raise NotImplemented
        if len(self.node_entity_names) < self.max_entities + self.max_ground_truth + self.max_objectives:
            self.node_entity_names += [[]] * (self.max_entities + self.max_ground_truth +
                                              self.max_objectives-len(self.node_entity_names))
        elif len(self.node_entity_names) == self.max_entities + self.max_ground_truth + self.max_objectives:
            pass
        else:
            raise NotImplemented

        return graphs

    def _initialize_node_entities(self):
        for graph in self.proof.entities+self.proof.ground_truth+self.proof.objectives:
            self._traverse_graph(graph)

    def _traverse_graph(self, graph):
        if isinstance(graph, Entity):
            if graph.name in self.node_entities:
                pass
            else:
                self.node_entities[graph.name] = (len(self.node_entities), deepcopy(graph))
            if graph.rnc_operands is None:
                pass
            else:
                for entity in graph.rnc_operands:
                    self._traverse_graph(entity)
        elif isinstance(graph, LogicStatement):
            for entity in graph.entities:
                self._traverse_graph(entity)
        else:
            raise NotImplemented

    def reset(self):
        self.proof = initialize_proof(
            proof_dir=self.proof_dir,
            np_random=self.np_random,
            total_configuration=self.env_config["max_configuration"]
        )
        self._initialize_node_entities()
        assert self.proof.entities[0].name == "NOOP"
        return self._get_obs()

    def render(self, mode='human'):
        raise NotImplemented


if __name__ == "__main__":
    from pprint import pprint
    import gym
    env = gym.make("TheoremProver-v0")

    from Inequality.logic.utils import standard_logic_functions
    x = Entity(name="input1")
    y = Entity(name="input2")
    z = Entity(name="input3")
    objectives = [standard_logic_functions["BiggerOrEqual"].execute_lf([x, z])]
    env.proof.objectives = objectives
    #
    # graphs, node_types = env._get_obs()
    # print(len(env.proof.entities+env.proof.ground_truth+env.proof.objectives), len(graphs))
    # for entity, graph in zip(env.proof.entities+env.proof.ground_truth+env.proof.objectives, graphs):
    #     print(entity.name)
    #     pprint(graph)

    env.proof.entities[0].name = "NOOP"
    lemma_index = [0]
    entity_indices = [2, 0, 0, 0, 0]
    action = np.array(lemma_index + entity_indices)
    pprint(env.step(action))
    print(env.choose_node(5, 1))
    #
    # pprint(env.node_entities)
    # print(len(env.node_entities))
