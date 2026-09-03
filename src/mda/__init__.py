"""Model Discovery Agent: LLM proposals, SMC³ inference, and Bayesian design."""

from .acquisition import joint_eig, model_eig, task_value_of_information
from .agent import AgentConfig, ModelDiscoveryAgent
from .expressions import SymbolicRegressionProposer, compile_expression
from .likelihoods import BootstrapParticleFilter, DeterministicGaussianLikelihood
from .neuronbench_adapter import NeuronBenchEnvironment, NeuronBenchProposer, NeuronHypothesis
from .openai_client import OpenAIJSONClient
from .posterior import BeliefState, ModelFit, fit_model_posterior
from .proposer import ModelProposer
from .smc import SMCConfig, parameter_posterior
from .types import CandidateModel, Datum, StateSpaceModel, UniformPrior

__all__ = [
    "AgentConfig",
    "BeliefState",
    "BootstrapParticleFilter",
    "CandidateModel",
    "Datum",
    "DeterministicGaussianLikelihood",
    "ModelDiscoveryAgent",
    "ModelFit",
    "ModelProposer",
    "NeuronBenchEnvironment",
    "NeuronBenchProposer",
    "NeuronHypothesis",
    "OpenAIJSONClient",
    "SMCConfig",
    "StateSpaceModel",
    "SymbolicRegressionProposer",
    "UniformPrior",
    "compile_expression",
    "fit_model_posterior",
    "joint_eig",
    "model_eig",
    "parameter_posterior",
    "task_value_of_information",
]
