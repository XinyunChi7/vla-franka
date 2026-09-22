#!/usr/bin/env python3
import unittest

from gripper_debouncer import GripperDebouncer


class GripperDebouncerTest(unittest.TestCase):
    def make(self):
        return GripperDebouncer(
            close_confirm_ticks=3,
            open_confirm_ticks=6,
        )

    def test_close_is_confirmed_but_not_immediate(self):
        gate = self.make()
        gate.reset(is_closed=False)
        self.assertEqual([gate.update(-1.0) for _ in range(3)], [None, None, True])

    def test_alternating_sign_never_switches(self):
        gate = self.make()
        gate.reset(is_closed=False)
        values = [-1.0, 1.0] * 10
        self.assertTrue(all(gate.update(value) is None for value in values))
        self.assertFalse(gate.stable_should_close)

    def test_small_negative_values_still_close(self):
        gate = self.make()
        gate.reset(is_closed=False)
        self.assertEqual([gate.update(-0.01) for _ in range(3)], [None, None, True])

    def test_open_is_slower_but_not_reverse_locked(self):
        gate = self.make()
        gate.reset(is_closed=False)
        for _ in range(3):
            edge = gate.update(-1.0)
        self.assertTrue(edge)
        self.assertEqual(
            [gate.update(1.0) for _ in range(6)],
            [None, None, None, None, None, False],
        )

    def test_same_state_never_republishes(self):
        gate = self.make()
        gate.reset(is_closed=True)
        self.assertTrue(all(gate.update(-1.0) is None for _ in range(30)))

    def test_one_tick_mode_disables_debounce(self):
        gate = GripperDebouncer(close_confirm_ticks=1, open_confirm_ticks=1)
        gate.reset(is_closed=False)
        self.assertIs(gate.update(-0.01), True)
        self.assertIs(gate.update(0.01), False)


if __name__ == "__main__":
    unittest.main()
