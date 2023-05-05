from common.policies import (
    RecurrentMaskableActorCriticCnnPolicy,
    RecurrentMaskableActorCriticPolicy,
    RecurrentMaskableMultiInputActorCriticPolicy,
)

MlpLstmPolicy = RecurrentMaskableActorCriticPolicy
CnnLstmPolicy = RecurrentMaskableActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMaskableMultiInputActorCriticPolicy
