
import copy

configs = dict()

config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=-1e-3,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=32,
        epochs=10,
        normalize_advantage=True,
    ),

    env=dict(id="Bolt"),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps = 2048* 10
    ),

    sampler=dict(
        batch_T=2048,
        batch_B=1,
        max_decorrelation_steps=0,
    )
)

configs["vime_1M_serial"] = config

config = copy.deepcopy(configs["vime_1M_serial"])
config["sampler"]["batch_B"] = 8
config["sampler"]["batch_T"] = 256
configs["vime_1M_cpu"] = config

config["algo"]["minibatches"] = 1
config["algo"]["epochs"] = 32
configs["ppo_32ep_1mb"] = config

