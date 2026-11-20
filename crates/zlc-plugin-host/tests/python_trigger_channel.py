"""The camera measurement's Trigger by channel switch, as its draft is told it: off, frames are counted and no
sequencer is asked for; on, the lanes are the trigger source's to state, each offered by its raw lane and then
the port's name, a lane must be chosen, and a cycle's size is that lane's triggers (shown, held, in Frames per
cycle). A camera that does not stamp its frames is not offered the switch."""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.nodes.camera.authoring import CAMERA_MEASUREMENT_SCHEMA, TRIGGER_CHANNEL, TRIGGER_ON, TRIGGER_SOURCE, TRIGGERED
from zlc_atom.nodes.camera.measurement import CAMERA_FRAMES_OUTPUT
from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE, _availability, _defaults, _describe_frames, _trigger_choices
from zlc_plugin.host import DeviceFacts

LANES = [{"lane": "ch11", "port": "emCCD", "label": "emCCD", "frames": 3}, {"lane": "ch04", "port": "repump", "label": "repump", "frames": None}]


def devices(*, stamps=True, lanes=LANES):
    known = {"camera": types.SimpleNamespace(stamps_frames=stamps)}
    if lanes is not None:
        known[TRIGGER_SOURCE] = DeviceFacts({"instance": "pulse_streamer", "role": "sequencer", "facts": {"trigger_lanes": lanes}})
    return known


def field(name):
    return next(item for item in CAMERA_MEASUREMENT_SCHEMA.fields if item.name == name)


class TriggerChannel(unittest.TestCase):
    def test_a_lane_is_offered_by_its_raw_name_then_the_ports(self):
        choices = _trigger_choices({}, None, devices())[TRIGGER_CHANNEL]
        self.assertEqual(choices, [["ch11", "ch11 · emCCD"], ["ch04", "ch04 · repump"]])
        self.assertEqual(_trigger_choices({}, None, devices(lanes=None)), {TRIGGER_CHANNEL: []})

    def test_whether_the_switch_is_offered_is_the_camera_s_to_say(self):
        self.assertNotIn(TRIGGERED, _availability(devices()))
        self.assertIn("does not stamp its frames", _availability(devices(stamps=False))[TRIGGERED])
        # Off asks nothing of any sequencer; on, a missing one is the draft's issue, not the switch's.
        self.assertNotIn(TRIGGERED, _availability(devices(lanes=None)))

    def test_a_cycle_is_the_chosen_lanes_triggers_while_the_switch_is_on(self):
        name = CAMERA_FRAMES_OUTPUT.name
        counted = _describe_frames({"frames_per_cycle": 2}, name, None, devices())
        self.assertEqual(counted[1], ("0", "1"), "off: as many as counted")
        kept = _describe_frames({"frames_per_cycle": 2, TRIGGERED: False, TRIGGER_CHANNEL: "ch11"}, name, None, devices())
        self.assertEqual(kept[1], ("0", "1"), "off: a lane kept in the draft is not used")
        chosen = _describe_frames({"frames_per_cycle": 2, TRIGGERED: True, TRIGGER_CHANNEL: "ch11"}, name, None, devices())
        self.assertEqual(chosen[1], ("0", "1", "2"), "on: the lane's three triggers, whatever was counted")
        self.assertIsNone(_describe_frames({"frames_per_cycle": 2, TRIGGERED: True, TRIGGER_CHANNEL: "ch04"}, name, None, devices()),
                          "no pulse says how many")

    def test_frames_per_cycle_shows_the_lanes_count_while_the_switch_is_on(self):
        self.assertEqual(_defaults({TRIGGERED: True, TRIGGER_CHANNEL: "ch11"}, {}, devices()), {"frames_per_cycle": 3})
        self.assertEqual(_defaults({TRIGGERED: True, TRIGGER_CHANNEL: "ch04"}, {}, devices()), {}, "no pulse: the typed count stays shown")
        self.assertEqual(_defaults({TRIGGERED: False, TRIGGER_CHANNEL: "ch11"}, {}, devices()), {})

    def test_the_sequencer_is_read_never_claimed_and_asked_for_only_while_the_switch_is_on(self):
        source = next(item for item in LOGIC_NODE.device_requirements if item.argument_name == TRIGGER_SOURCE)
        self.assertTrue(source.reads_only)
        self.assertEqual(source.enabled_when, TRIGGER_ON)
        self.assertEqual(source.capability_token, "trigger.source")
        self.assertEqual(field("frames_per_cycle").enabled_when, (TRIGGERED, (False,)))
        channel = field(TRIGGER_CHANNEL)
        self.assertEqual(channel.enabled_when, TRIGGER_ON)
        self.assertTrue(channel.required)


if __name__ == "__main__":
    unittest.main()
