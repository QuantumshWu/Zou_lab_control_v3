"""The camera measurement's Trigger channel, as its draft is told it: the lanes are the trigger source's
to state, each offered by its raw lane and then the port's name; a camera that does not stamp its frames
is not offered it; a cycle's size is the chosen lane's while one is chosen."""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.nodes.camera.authoring import CAMERA_MEASUREMENT_SCHEMA, TRIGGER_CHANNEL, TRIGGER_SOURCE
from zlc_atom.nodes.camera.measurement import CAMERA_FRAMES_OUTPUT
from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE, _availability, _describe_frames, _trigger_choices
from zlc_plugin.host import DeviceFacts

LANES = [{"lane": "ch11", "port": "emCCD", "label": "emCCD", "frames": 3}, {"lane": "ch04", "port": "repump", "label": "repump", "frames": None}]


def devices(*, stamps=True, lanes=LANES):
    known = {"camera": types.SimpleNamespace(stamps_frames=stamps)}
    if lanes is not None:
        known[TRIGGER_SOURCE] = DeviceFacts({"instance": "pulse_streamer", "role": "sequencer", "facts": {"trigger_lanes": lanes}})
    return known


class TriggerChannel(unittest.TestCase):
    def test_a_lane_is_offered_by_its_raw_name_then_the_ports(self):
        choices = _trigger_choices({}, None, devices())[TRIGGER_CHANNEL]
        self.assertEqual(choices, [["ch11", "ch11 · emCCD"], ["ch04", "ch04 · repump"]])
        self.assertEqual(_trigger_choices({}, None, devices(lanes=None)), {TRIGGER_CHANNEL: []})

    def test_whether_it_is_offered_is_the_camera_s_and_the_sequencer_s_to_say(self):
        self.assertNotIn(TRIGGER_CHANNEL, _availability(devices()))
        self.assertIn("does not stamp its frames", _availability(devices(stamps=False))[TRIGGER_CHANNEL])
        self.assertIn("no sequencer is open", _availability(devices(lanes=None))[TRIGGER_CHANNEL])

    def test_a_cycle_is_the_chosen_lanes_triggers_and_unknown_while_it_plays_nothing(self):
        name = CAMERA_FRAMES_OUTPUT.name
        counted = _describe_frames({"frames_per_cycle": 2}, name, None, devices())
        self.assertEqual(counted[1], ("0", "1"), "no lane chosen: as many as counted")
        chosen = _describe_frames({"frames_per_cycle": 2, TRIGGER_CHANNEL: "ch11"}, name, None, devices())
        self.assertEqual(chosen[1], ("0", "1", "2"), "the lane's three triggers, whatever was counted")
        self.assertIsNone(_describe_frames({"frames_per_cycle": 2, TRIGGER_CHANNEL: "ch04"}, name, None, devices()), "no pulse says how many")

    def test_the_sequencer_is_read_never_claimed_and_the_count_is_shut_while_a_lane_is_chosen(self):
        source = next(item for item in LOGIC_NODE.device_requirements if item.argument_name == TRIGGER_SOURCE)
        self.assertTrue(source.reads_only and source.optional)
        self.assertEqual(source.capability_token, "trigger.source")
        frames = next(field for field in CAMERA_MEASUREMENT_SCHEMA.fields if field.name == "frames_per_cycle")
        self.assertEqual(frames.enabled_when, (TRIGGER_CHANNEL, (None, "")))


if __name__ == "__main__":
    unittest.main()
