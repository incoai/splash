"""CPU checks of XML references and overlap accounting used for trace claims."""

import tempfile
import unittest
from pathlib import Path

from dev.tools.analyze_flash_metal_trace import iter_rows, merge_intervals, union_ns


class MetalTraceExportTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "export.xml"
            path.write_text(text)
            return list(iter_rows(path))

    def test_nested_process_definition_before_process_reference(self):
        rows = self.parse("""<trace-query-result><node><schema>
          <col><mnemonic>label</mnemonic></col><col><mnemonic>process</mnemonic></col>
          <col><mnemonic>cmdbuffer-id</mnemonic></col></schema>
          <row><formatted-label id="1" fmt="Work (native (39736))">
            <process id="2" fmt="native (39736)"><pid id="3">39736</pid></process>
          </formatted-label><process ref="2"/>
          <metal-command-buffer-id id="4" fmt="0x7b">123</metal-command-buffer-id></row>
          <row><formatted-label ref="1"/><process ref="2"/><metal-command-buffer-id ref="4"/></row>
          </node></trace-query-result>""")
        self.assertEqual([row["process"].pid for row in rows], [39736, 39736])
        self.assertEqual([row["cmdbuffer-id"].value for row in rows], [123, 123])

    def test_ids_are_not_shared_between_exports(self):
        template = """<node><schema><col><mnemonic>process</mnemonic></col></schema>
          <row><process id="1" fmt="native ({pid})"><pid id="2">{pid}</pid></process></row></node>"""
        self.assertEqual(self.parse(template.format(pid=10))[0]["process"].pid, 10)
        self.assertEqual(self.parse(template.format(pid=20))[0]["process"].pid, 20)

    def test_duplicate_tag_types_use_schema_position(self):
        rows = self.parse("""<node><schema><col><mnemonic>category</mnemonic></col>
          <col><mnemonic>event</mnemonic></col></schema><row>
          <gpu-driver-name id="1">Driver Processing</gpu-driver-name>
          <gpu-driver-name id="2">Wire Memory</gpu-driver-name></row></node>""")
        self.assertEqual(rows[0]["category"].value, "Driver Processing")
        self.assertEqual(rows[0]["event"].value, "Wire Memory")

    def test_missing_reference_fails(self):
        with self.assertRaises(ValueError):
            self.parse("""<node><schema><col><mnemonic>start</mnemonic></col></schema>
              <row><start-time ref="100"/></row></node>""")

    def test_ns_are_not_formatted_microseconds(self):
        rows = self.parse("""<node><schema><col><mnemonic>duration</mnemonic></col></schema>
          <row><duration id="1" fmt="62.75 µs">62750</duration></row></node>""")
        self.assertEqual(rows[0]["duration"].value, 62750)

    def test_nested_and_overlapping_intervals_count_once(self):
        self.assertEqual(
            merge_intervals([(0, 10), (2, 8), (5, 15), (20, 25)]), [(0, 15), (20, 25)]
        )
        self.assertEqual(union_ns([(0, 10), (2, 8), (5, 15), (20, 25)]), 20)

    def test_adjacent_empty_and_reversed_intervals(self):
        self.assertEqual(merge_intervals([(5, 5), (10, 8), (0, 4), (4, 7)]), [(0, 7)])
        self.assertEqual(union_ns([]), 0)


if __name__ == "__main__":
    unittest.main()
