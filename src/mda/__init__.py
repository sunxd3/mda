"""Model Discovery Agent: LLM proposals, approximate Bayesian inference, and design."""

from .acquisition import (
    full_observation_task_value_of_information,
    joint_eig, model_eig, task_value_of_information,
)
from .agent import AgentConfig, ModelDiscoveryAgent
from .expressions import SymbolicRegressionProposer, compile_expression
from .evidence import BICEvidence, RecursiveEvidence, TemperedSMCEvidence
from .likelihoods import (
    BayesianSyntheticLikelihood, BootstrapParticleFilter, DeterministicGaussianLikelihood,
)
from .neuronbench_adapter import NeuronBenchEnvironment, NeuronBenchProposer, NeuronHypothesis
from .openai_client import OpenAIJSONClient
from .optimizers import CMAESOptimizer
from .posterior import BeliefState, ModelFit, fit_model_posterior
from .proposer import ModelProposer
from .smc import SMCConfig, parameter_posterior
from .types import CandidateModel, Datum, StateSpaceModel, UniformPrior

__all__ = [
    "AgentConfig",
    "BICEvidence",
    "BayesianSyntheticLikelihood",
    "BeliefState",
    "BootstrapParticleFilter",
    "CMAESOptimizer",
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
    "RecursiveEvidence",
    "SMCConfig",
    "StateSpaceModel",
    "SymbolicRegressionProposer",
    "TemperedSMCEvidence",
    "UniformPrior",
    "compile_expression",
    "fit_model_posterior",
    "full_observation_task_value_of_information",
    "joint_eig",
    "model_eig",
    "parameter_posterior",
    "task_value_of_information",
]
