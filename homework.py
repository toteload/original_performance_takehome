"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.cycle_instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add2(self, engine, slot):
        self.cycle_instrs.append((engine, slot))

    def end_cycle(self):
        instrs = defaultdict(list)
        for engine, slot in self.cycle_instrs:
            instrs[engine].append(slot)
        self.instrs.append(instrs)
        self.cycle_instrs = []

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
            "ignore_me_pls",
        ]

        # Reserve scratch memory for all the init_vars
        for v in init_vars:
            self.alloc_scratch(v, 1)

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # for all the init_vars:
        #   1. load the index into tmp1
        #   2. read memory at index into the corresponding init_var
        #for i, v in enumerate(init_vars):
        #    self.add("load", ("const", tmp1, i))
        #    self.add("load", ("load", self.scratch[v], tmp1))

        self.add("load", ("vload", self.scratch["rounds"], zero_const))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        indices_p_i = self.alloc_scratch("indices_p_i");
        values_p_i = self.alloc_scratch("values_p_i");

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                # val = mem[inp_values_p + i]
                self.add2(*("alu", ("+", indices_p_i, self.scratch["inp_indices_p"], i_const)))
                self.add2(*("alu", ("+", values_p_i, self.scratch["inp_values_p"], i_const)))
                self.end_cycle()
                self.add2(*("load", ("load", tmp_idx, indices_p_i)))
                self.add2(*("load", ("load", tmp_val, values_p_i)))
                self.end_cycle()

                #self.add(*("debug", ("compare", tmp_idx, (round, i, "idx"))))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "val"))))

                # node_val = mem[forest_values_p + idx]
                self.add(*("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                self.add(*("load", ("load", tmp_node_val, tmp_addr)))

                #self.add(*("debug", ("compare", tmp_node_val, (round, i, "node_val"))))

                # val = myhash(val ^ node_val)
                self.add(*("alu", ("^", tmp_val, tmp_val, tmp_node_val)))

                # val = (val << 12) + (val + 0x7ED55D16)
                # val = val * 4097 + 0x7ED55D16
                self.add2(*("alu", ("+", tmp1, tmp_val, self.scratch_const(0x7ED55D16))))
                self.add2(*("alu", ("<<", tmp2, tmp_val, self.scratch_const(12))))
                self.end_cycle()
                self.add(*("alu", ("+", tmp_val, tmp1, tmp2)))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "hash_stage", 0))))

                self.add2(*("alu", ("^", tmp1, tmp_val, self.scratch_const(0xC761C23C))))
                self.add2(*("alu", (">>", tmp2, tmp_val, self.scratch_const(19))))
                self.end_cycle()
                self.add(*("alu", ("^", tmp_val, tmp1, tmp2)))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "hash_stage", 1))))

                # val = (val << 5) + (val + 0x165667B1)
                # val = val * 33 + 0x165667B1
                self.add2(*("alu", ("+", tmp1, tmp_val, self.scratch_const(0x165667B1))))
                self.add2(*("alu", ("<<", tmp2, tmp_val, self.scratch_const(5))))
                self.end_cycle()
                self.add(*("alu", ("+", tmp_val, tmp1, tmp2)))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "hash_stage", 2))))

                # val = (val << 9) + (val + 0xD3A2646C)
                # val = val * 513 + 0xD3A2646C
                self.add2(*("alu", ("+", tmp1, tmp_val, self.scratch_const(0xD3A2646C))))
                self.add2(*("alu", ("<<", tmp2, tmp_val, self.scratch_const(9))))
                self.end_cycle()
                self.add(*("alu", ("^", tmp_val, tmp1, tmp2)))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "hash_stage", 3))))

                # val = (val << 3) + (val + 0xFD7046C5)
                # val = val * 9 + 0xFD7046C5
                self.add2(*("alu", ("+", tmp1, tmp_val, self.scratch_const(0xFD7046C5))))
                self.add2(*("alu", ("<<", tmp2, tmp_val, self.scratch_const(3))))
                self.end_cycle()
                self.add(*("alu", ("+", tmp_val, tmp1, tmp2)))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "hash_stage", 4))))

                self.add2(*("alu", ("^", tmp1, tmp_val, self.scratch_const(0xB55A4F09))))
                self.add2(*("alu", (">>", tmp2, tmp_val, self.scratch_const(16))))
                self.end_cycle()
                self.add(*("alu", ("^", tmp_val, tmp1, tmp2)))
                #self.add(*("debug", ("compare", tmp_val, (round, i, "hash_stage", 5))))

                #self.add(*("debug", ("compare", tmp_val, (round, i, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # idx = 2*idx + (0 if val % 2 == 0 else 1) + 1
                # idx = (2*idx) + (val & 1) + 1
                self.add2("alu", ("*", tmp_idx, tmp_idx, two_const))
                self.add2("alu", ("&", tmp1, tmp_val, one_const))
                self.end_cycle()
                self.add("alu", ("+", tmp_idx, tmp1, tmp_idx))
                self.add("alu", ("+", tmp_idx, tmp_idx, one_const))

                # idx = 0 if idx >= n_nodes else idx
                self.add(*("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                self.add(*("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))

                # mem[inp_indices_p + i] = idx
                # mem[inp_values_p + i] = val
                self.add2(*("store", ("store", indices_p_i, tmp_idx)))
                self.add2(*("store", ("store", values_p_i, tmp_val)))
                self.end_cycle()

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    def test_kernel_correctness(self):
        for batch in range(1, 3):
            for forest_height in range(3):
                do_kernel_test(
                    forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
                )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
