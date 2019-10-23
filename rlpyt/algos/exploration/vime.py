import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])

class VIME(PolicyGradientAlgo):

    def __init__(
            self,
            OptimCls,
            optimizer_args=None,
            step_size=1e-2):
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

    def optimize_agent(self, itr, samples):
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        return_, advantage, valid = self.process_returns(samples)

        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=samples, agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )

        if recurrent:
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*[] for _ in range(len(OptInfo._fields)))

        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_rate = init_rnn_state[B_idxs] if recurrent else None

                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())

        return opt_info
    
    # def loss(self, agent_inputs, action,
    #          return_, advantage, valid,
    #          old_dist_info, init_rnn_state):
    #     if 



        