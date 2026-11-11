"""An arena keeps the free blocks its work comes back for and gives the rest back: what stayed free
through a whole turn of its size, and a size that has gone unused for longer than it ever did. A
borrower lets a mapping go with its last view."""
import gc
import pathlib
import sys
import unittest
import uuid

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_plugin.blocks import GRANULARITY, Arena, Attachments

FRAME = (GRANULARITY * 2,)   # a 128 KiB block
OTHER = (GRANULARITY * 5,)


class Bench:
    def __init__(self):
        self.retired = []
        self.arena = Arena(uuid.uuid4().hex, retired=self.retired.extend)

    def hold(self, shape, count):
        """``count`` allocations of ``shape``, each lent once (as frames kept by the core are)."""
        held = []
        for _ in range(count):
            array = self.arena.allocate(shape, "u1")
            reference = self.arena.lend(array, None)
            held.append(reference["block"])
            del array
        gc.collect()
        return held

    def release(self, names):
        for name in names:
            self.arena.released(name)

    @property
    def mapped(self):
        return self.arena.mapped_bytes


class GiveBack(unittest.TestCase):
    def test_a_block_that_keeps_coming_back_is_kept(self):
        bench = Bench()
        for _ in range(50):
            bench.release(bench.hold(FRAME, 1))
        self.assertEqual((bench.mapped, bench.retired), (FRAME[0], []), "one block, recycled every time")
        # A batch taken and returned whole, over and over: every block of it is needed by each turn.
        for _ in range(6):
            bench.release(bench.hold(FRAME, 16))
        self.assertEqual((bench.mapped, bench.retired), (16 * FRAME[0], []))

    def test_what_a_retention_no_longer_holds_is_given_back_within_a_turn(self):
        bench = Bench()
        ring = bench.hold(FRAME, 400)
        self.assertEqual(bench.mapped, 400 * FRAME[0])
        # Steady: the oldest is released as each new one is taken.
        for _ in range(800):
            ring += bench.hold(FRAME, 1)
            bench.release([ring.pop(0)])
        self.assertLessEqual(bench.mapped, 401 * FRAME[0], "a full ring keeps what it turns over, no more")
        self.assertEqual(bench.retired, [])
        # The retention ends: everything is released, and the work goes on with one frame at a time.
        bench.release(ring)
        for _ in range(410):
            bench.release(bench.hold(FRAME, 1))
        self.assertLessEqual(bench.mapped, 2 * FRAME[0], "given back after one turn of the blocks it had")
        self.assertGreaterEqual(len(bench.retired), 399)
        self.assertEqual(len(set(bench.retired)), len(bench.retired), "each block is given back once")

    def test_a_size_no_longer_allocated_is_given_back_and_one_used_in_turn_is_kept(self):
        bench = Bench()
        # Two sizes in every cycle, one of them four times: neither is idle for longer than its usual.
        for _ in range(20):
            bench.release(bench.hold(FRAME, 1))
            for _ in range(4):
                bench.release(bench.hold(OTHER, 1))
        learned = len(bench.retired)
        self.assertLessEqual(learned, 1, "at most once, while its usual gap was not yet known")
        for _ in range(20):
            bench.release(bench.hold(FRAME, 1))
            for _ in range(4):
                bench.release(bench.hold(OTHER, 1))
        self.assertEqual(len(bench.retired), learned, "a size used in every cycle keeps its block")
        # The frame size changes (a new ROI): the old size's blocks go.
        for _ in range(12):
            bench.release(bench.hold(OTHER, 1))
        self.assertEqual(bench.mapped, OTHER[0], "only the size still in use is mapped")

    def test_a_borrower_unmaps_a_block_with_its_last_view(self):
        bench = Bench()
        released = []
        borrower = Attachments(released.append)
        array = bench.arena.allocate(FRAME, "u1")
        array[:] = 7
        reference = bench.arena.lend(array, None)
        first = borrower.attach(reference)
        second = borrower.attach(reference)
        self.assertEqual((int(first[0]), borrower.mapped_blocks), (7, 1))
        del first
        gc.collect()
        other = bench.arena.lend(bench.arena.allocate(OTHER, "u1"), None)
        kept = borrower.attach(other)
        self.assertEqual(borrower.mapped_blocks, 2, "a view of it is still alive")
        del second
        gc.collect()
        again = borrower.attach(other)
        self.assertEqual(borrower.mapped_blocks, 1, "its last view gone, the mapping went at the next attach")
        self.assertEqual(released.count(reference["block"]), 2)
        del kept, again


if __name__ == "__main__": unittest.main()
