
import time
import multiprocessing as mp
import psutil
import torch
from collections import deque
import math

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.utils.synchronize import drain_queue, find_port


THROTTLE_WAIT = 0.05


class AsyncRlBase(BaseRunner):

    _eval = False

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            affinity,
            updates_per_sync=1,
            seed=None,
            log_interval_steps=1e5,
            ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        save__init__args(locals())

    def train(self):
        throttle_itr, delta_throttle_itr = self.startup()
        throttle_time = 0.
        sampler_itr = itr = 0
        if self._eval:
            while self.ctrl.sampler_itr.value < 1:  # Sampler does eval first.
                time.sleep(THROTTLE_WAIT)
            traj_infos = drain_queue(self.traj_infos_queue)
            self.store_diagnostics(0, 0, traj_infos, ())
            self.log_diagnostics(0, 0, 0)
        log_counter = 0
        while True:  # Run until sampler hits n_steps and sets ctrl.quit=True.
            with logger.prefix(f"opt_itr #{itr} "):
                while self.ctrl.sampler_itr.value < throttle_itr:
                    if self.ctrl.quit.value:
                        break
                    time.sleep(THROTTLE_WAIT)
                    throttle_time += THROTTLE_WAIT
                if self.ctrl.quit.value:
                    break
                if self.ctrl.opt_throttle is not None:
                    self.ctrl.opt_throttle.wait()
                throttle_itr += delta_throttle_itr
                opt_info = self.algo.optimize_agent(itr,
                    sampler_itr=self.ctrl.sampler_itr.value)
                self.agent.send_shared_memory()  # To sampler.
                with self.ctrl.sampler_itr.get_lock():
                    # Lock to prevent traj_infos splitting across itr.
                    traj_infos = drain_queue(self.traj_infos_queue)
                    sampler_itr = self.ctrl.sampler_itr.value
                self.store_diagnostics(itr, sampler_itr, traj_infos, opt_info)
                if (sampler_itr // self.log_interval_itrs > log_counter):
                    self.log_diagnostics(itr, sampler_itr, throttle_time)
                    log_counter += 1
                    throttle_time = 0.
            itr += 1
        self.shutdown()

    def startup(self):
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        double_buffer, examples = self.sampler.master_runner_initialize(
            agent=self.agent,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            seed=self.seed,
        )
        replay_buffer = self.algo.initialize_replay_buffer(
            batch_spec=self.sampler.batch_spec,
            examples=examples,
            mid_batch_reset=self.sampler.mid_batch_reset,
            async_=True,
            updates_per_sync=self.updates_per_sync,
            agent=self.agent,
        )
        self.sampler_batch_size = self.sampler.batch_spec.size
        self.world_size = len(self.affinity.optimizer)
        n_itr = self.get_n_itr()  # Number of sampler iterations.
        self.traj_infos_queue = mp.Queue()
        self.launch_workers(n_itr, double_buffer, replay_buffer)
        main_affinity = self.affinity.optimizer[0]
        p = psutil.Process()
        p.cpu_affinity(main_affinity["cpus"])
        logger.log(f"Optimizer master CPU affinity: {p.cpu_affinity()}.")
        torch.set_num_threads(main_affinity["torch_threads"])
        logger.log(f"Optimizer master Torch threads: {torch.get_num_threads()}.")
        self.agent.initialize_device(
            cuda_idx=main_affinity.get("cuda_idx", None),
            ddp=self.world_size > 1,
        )
        self.algo.async_initialize(
            # agent=self.agent,
            sampler_n_itr=n_itr,
            rank=0,
            world_size=self.world_size,
        )
        throttle_itr = 1 + getattr(self.algo,
            "min_steps_learn", 0) // self.sampler_batch_size
        delta_throttle_itr = (self.algo.batch_size * self.world_size *
            self.algo.updates_per_optimize /  # (is updates_per_sync)
            (self.sampler_batch_size * self.algo.replay_ratio))
        self.initialize_logging()
        return throttle_itr, delta_throttle_itr

    def launch_workers(self, n_itr, double_buffer, replay_buffer):
        self.ctrl = self.build_ctrl(self.world_size)
        self.launch_sampler(n_itr)
        self.launch_memcpy(double_buffer, replay_buffer)
        self.launch_optimizer_workers(n_itr)

    def get_n_itr(self):
        log_interval_itrs = max(self.log_interval_steps //
            self.sampler_batch_size, 1)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} sampler iterations.")
        return n_itr

    def build_ctrl(self, world_size):
        opt_throttle = (mp.Barrier(world_size) if world_size > 1 else
            None)
        return AttrDict(
            quit=mp.Value('b', lock=True),
            quit_opt=mp.RawValue('b'),
            sample_ready=[mp.Semaphore(0) for _ in range(2)],  # Double buffer.
            sample_copied=[mp.Semaphore(1) for _ in range(2)],
            sampler_itr=mp.Value('l', lock=True),
            opt_throttle=opt_throttle,
            eval_time=mp.Value('d', lock=True),
            )

    def launch_optimizer_workers(self, n_itr):
        if self.world_size == 1:
            return
        offset = self.affinity.optimizer[0].get("master_cpus", [0])[0]
        port = find_port(offset=offset)
        affinities = self.affinity.optimizer
        runners = [AsyncOptWorker(
            rank=rank,
            world_size=self.world_size,
            algo=self.algo,
            agent=self.agent,
            n_itr=n_itr,
            affinity=affinities[rank],
            seed=self.seed + 100,
            ctrl=self.ctrl,
            port=port,
            ) for rank in range(1, len(affinities))]
        procs = [mp.Process(target=r.optimize, args=()) for r in runners]
        for p in procs:
            p.start()
        torch.distributed.init_process_group(
            backend="nccl",
            rank=0,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )
        self.optimizer_procs = procs

    def launch_memcpy(self, sample_buffers, replay_buffer):
        procs = list()
        for i in range(len(sample_buffers)):  # (2 for double-buffer.)
            ctrl = AttrDict(
                quit=self.ctrl.quit,
                sample_ready=self.ctrl.sample_ready[i],
                sample_copied=self.ctrl.sample_copied[i],
            )
            procs.append(mp.Process(target=memory_copier,
                args=(sample_buffers[i], self.algo.samples_to_buffer,
                replay_buffer, ctrl)))
        for p in procs:
            p.start()
        self.memcpy_procs = procs

    def launch_sampler(self, n_itr):
        target = run_async_sampler
        kwargs = dict(
            sampler=self.sampler,
            affinity=self.affinity.sampler,
            ctrl=self.ctrl,
            traj_infos_queue=self.traj_infos_queue,
            n_itr=n_itr,
        )
        if self._eval:
            target = run_async_sampler_eval
            kwargs["eval_itrs"] = self.log_interval_itrs
        self.sampler_proc = mp.Process(target=target, kwargs=kwargs)
        self.sampler_proc.start()

    def shutdown(self):
        self.pbar.stop()
        logger.log("Master optimizer shutting down, joining sampler process...")
        self.sampler_proc.join()
        logger.log("Joining memory copiers...")
        for p in self.memcpy_procs:
            p.join()
        if self.ctrl.opt_throttle is not None:
            logger.log("Joining optimizer processes...")
            self.ctrl.quit_opt.value = True
            self.ctrl.opt_throttle.wait()
            for p in self.optimizer_procs:
                p.join()
        logger.log("All processes shutdown.  Training complete.")

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._last_itr = 0

    def get_itr_snapshot(self, itr, sampler_itr):
        return dict(
            itr=itr,
            sampler_itr=sampler_itr,
            cum_steps=sampler_itr * self.sampler_batch_size,
            cum_updates=itr * self.algo.updates_per_optimize,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr, sample_itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr, sample_itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def get_traj_info_kwargs(self):
        return dict(discount=getattr(self.algo, "discount", 1))

    def store_diagnostics(self, itr, sampler_itr, traj_infos, opt_info):
        self._traj_infos.extend(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((sampler_itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, sampler_itr, throttle_time):
        self.pbar.stop()
        self.save_itr_snapshot(itr, sampler_itr)
        new_time = time.time()
        time_elapsed = new_time - self._last_time
        samples_per_second = (float('nan') if itr == 0 else
            self.log_interval_itrs * self.sampler_batch_size / time_elapsed)
        updates_per_second = (float('nan') if itr == 0 else
            self.algo.updates_per_optimize * (itr - self._last_itr) / time_elapsed)
        replay_ratio = updates_per_second * self.algo.batch_size / samples_per_second
        cum_steps = sampler_itr * self.sampler_batch_size
        cum_updates = itr * self.algo.updates_per_optimize
        cum_replay_ratio = cum_updates * self.algo.batch_size / max(1, cum_steps)
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('SamplerIteration', sampler_itr)
        logger.record_tabular('CumTime (s)', new_time - self._start_time)
        logger.record_tabular('CumSteps', cum_steps)
        logger.record_tabular('CumUpdates', cum_updates)
        logger.record_tabular('ReplayRatio', replay_ratio)
        logger.record_tabular('CumReplayRatio', cum_replay_ratio)
        logger.record_tabular('SamplesPerSecond', samples_per_second)
        logger.record_tabular('UpdatesPerSecond', updates_per_second)
        logger.record_tabular('OptThrottle', (time_elapsed - throttle_time) /
            time_elapsed)

        self._log_infos()
        self._last_time = new_time
        self._last_itr = itr
        logger.dump_tabular(with_prefix=False)
        logger.log(f"Optimizing over {self.log_interval_itrs} sampler "
            "iterations.")
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self, traj_infos=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k,
                        [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class AsyncRl(AsyncRlBase):

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._cum_completed_trajs = 0
        self._new_completed_trajs = 0
        super().initialize_logging()
        logger.log(f"Optimizing over {self.log_interval_itrs} sampler "
            "iterations.")
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, sampler_itr, traj_infos, opt_info):
        self._cum_completed_trajs += len(traj_infos)
        self._new_completed_trajs += len(traj_infos)
        super().store_diagnostics(itr, sampler_itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, sampler_itr, throttle_time):
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
            sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr, sampler_itr, throttle_time)
        self._new_completed_trajs = 0


class AsyncRlEval(AsyncRlBase):

    _eval = True

    def initialize_logging(self):
        self._traj_infos = list()
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def log_diagnostics(self, itr, sampler_itr, throttle_time):
        if not self._traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in self._traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(self._traj_infos))
        logger.record_tabular('CumEvalTime', self.ctrl.eval_time.value)
        super().log_diagnostics(itr, sampler_itr, throttle_time)
        self._traj_infos = list()  # Clear after each eval.
        self.pbar = ProgBarCounter(self.log_interval_itrs)


###############################################################################
# Worker processes.
###############################################################################


class AsyncOptWorker(object):

    def __init__(
            self,
            rank,
            world_size,
            algo,
            agent,
            n_itr,
            affinity,
            seed,
            ctrl,
            port
            ):
        save__init__args(locals())

    def optimize(self):
        self.startup()
        itr = 0
        while True:
            self.ctrl.opt_throttle.wait()
            if self.ctrl.quit_opt.value:
                break
            self.algo.optimize_agent(itr, sampler_itr=self.ctrl.sampler_itr.value)  # Leave un-logged.
            itr += 1
        self.shutdown()

    def startup(self):
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        print("INSIDE OPT WORKER STARTUP")
        p = psutil.Process()
        p.cpu_affinity(self.affinity["cpus"])
        logger.log(f"Optimizer rank {self.rank} CPU affinity: {p.cpu_affinity()}.")
        torch.set_num_threads(self.affinity["torch_threads"])
        logger.log(f"Optimizer rank {self.rank} Torch threads: {torch.get_num_threads()}.")
        logger.log(f"Optimizer rank {self.rank} CUDA index: "
            f"{self.affinity.get('cuda_idx', None)}.")
        set_seed(self.seed)
        self.agent.initialize_device(
            cuda_idx=self.affinity.get("cuda_idx", None),
            ddp=True,
        )
        self.algo.async_initialize(
            agent=self.agent,
            sampler_n_itr=self.n_itr,
            rank=self.rank,
            world_size=self.world_size,
        )

    def shutdown(self):
        logger.log(f"Async optimizaiton worker {self.rank} shutting down.")


def run_async_sampler(sampler, affinity, ctrl, traj_infos_queue, n_itr):
    sampler.sampler_process_initialize(affinity)
    j = 0
    for itr in range(n_itr):
        ctrl.sample_copied[j].acquire()
        traj_infos = sampler.obtain_samples(itr, j)
        ctrl.sample_ready[j].release()
        with ctrl.sampler_itr.get_lock():
            for traj_info in traj_infos:
                traj_infos_queue.put(traj_info)
            ctrl.sampler_itr.value = itr
        j ^= 1  # Double buffer.
    logger.log(f"Async sampler reached final itr: {itr}, quitting.")
    ctrl.quit.value = True  # This ends the experiment.
    sampler.shutdown()
    for s in ctrl.sample_ready:
        s.release()  # Let memcpy workers finish and quit.


def run_async_sampler_eval(sampler, affinity, ctrl, traj_infos_queue,
        n_itr, eval_itrs):
    sampler.sampler_process_initialize(affinity)
    j = 0
    for itr in range(n_itr):
        ctrl.sample_copied[j].acquire()
        sampler.obtain_samples(itr, j)
        ctrl.sample_ready[j].release()
        if itr % eval_itrs == 0:
            eval_time = -time.time()
            traj_infos = sampler.evaluate_agent(itr)
            eval_time += time.time()
            ctrl.eval_time.value += eval_time  # Not atomic but only writer.
            with ctrl.sampler_itr.get_lock():
                for traj_info in traj_infos:
                    traj_infos_queue.put(traj_info)
                ctrl.sampler_itr.value = itr
        else:
            ctrl.sampler_itr.value = itr
        j ^= 1  # Double buffer
    logger.log(f"Async sampler reached final itr: {itr}, quitting.")
    ctrl.quit.value = True  # This ends the experiment.
    sampler.shutdown()
    for s in ctrl.sample_ready:
        s.release()  # Let memcpy workers finish and quit.


def memory_copier(sample_buffer, samples_to_buffer, replay_buffer, ctrl):
    while True:
        ctrl.sample_ready.acquire()
        if ctrl.quit.value:
            break
        replay_buffer.append_samples(samples_to_buffer(sample_buffer))
        ctrl.sample_copied.release()
    logger.log("Memory copier shutting down.")
